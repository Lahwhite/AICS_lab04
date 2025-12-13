from dataclasses import dataclass
from typing import Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as np
from mindspore import ops
from mindspore.common.initializer import Normal, initializer
from mindspore.ops import functional as F

# TODO: 实现 GPT 模型结构


class LayerNorm(nn.Cell):
    """LayerNorm but with an optional bias."""

    pass


class CausalSelfAttention(nn.Cell):
    pass


class MLP(nn.Cell):
    pass


class Block(nn.Cell):
    pass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 5440 # padded up to nearest multiple of 64 for efficiency # fmt:skip
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster # fmt:skip


class GPT(nn.Cell):

    def __init__(self, config: GPTConfig):
        super().__init__()
        pass

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.get_parameters())
        if non_embedding:
            n_params -= self.wpe.embedding_table.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, ms.Parameter):
            module.set_data(initializer(Normal(mean=0.0, sigma=0.02), module.shape))
        elif isinstance(module, nn.Linear):
            module.weight.set_data(
                initializer(Normal(mean=0.0, sigma=0.02), module.weight.shape)
            )
            if module.bias is not None:
                module.bias.set_data(
                    initializer(Normal(mean=0.0, sigma=0.02), module.bias.shape)
                )
        elif isinstance(module, nn.Embedding):
            embedding_table = getattr(module, "embedding_table", None)
            if embedding_table is not None:
                embedding_table.set_data(
                    initializer(Normal(mean=0.0, sigma=0.02), embedding_table.shape)
                )

    def construct(self, idx: ms.Tensor, targets: Optional[ms.Tensor] = None):
        pass

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        def decay_filter(x: ms.Parameter) -> bool:
            return len(x.shape) >= 2

        params = self.trainable_params()
        decay_params = list(filter(decay_filter, params))
        other_params = list(filter(lambda x: not decay_filter(x), params))
        group_params = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": other_params, "weight_decay": 0.0},
            {"order_params": params},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in other_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with"
            f" {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(other_params)}, with"
            f" {num_nodecay_params:,} parameters"
        )
        optimizer = nn.AdamWeightDecay(
            group_params,
            learning_rate=learning_rate,
            beta1=betas[0],
            beta2=betas[1],
            weight_decay=0.0,
        )

        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        pass


class GPTWithLoss(nn.Cell):
    def __init__(self, gpt: GPT):
        super().__init__()
        self.gpt = gpt

    def construct(self, input_ids: ms.Tensor):
        tokens = input_ids[:, :-1]
        labels = input_ids[:, 1:]
        _, loss = self.gpt(tokens, labels)
        return loss


_grad_scale = ops.composite.MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.reciprocal(scale)


class GPTTrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """
    通用 GPT 训练封装：
    1. 包含混合精度 (FP16) 的 Loss Scaling
    2. 包含梯度裁剪 (Gradient Clipping) 防止 Loss NaN
    """

    def __init__(self, network, optimizer, scale_sense, gradient_clip=1.0):
        super(GPTTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.gradient_clip = gradient_clip
        self.hyper_map = ops.HyperMap()

    def construct(self, input_ids: ms.Tensor):
        weights = self.weights
        loss = self.network(input_ids)

        status, scaling_sens = self.start_overflow_check(loss, self.scale_sense)
        grads = self.grad(self.network, weights)(input_ids, scaling_sens)

        # 2. 梯度还原（Unscale）：把梯度除以 scale，变回真实大小
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)

        grads = ops.clip_by_global_norm(grads, self.gradient_clip)

        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)

        if not overflow:
            self.optimizer(grads)
            return loss
        else:
            print("Gradient overflow detected, skipping step and reducing loss scale.")
            return ops.zeros_like(loss)
