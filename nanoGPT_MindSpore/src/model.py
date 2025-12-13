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

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = ms.Parameter(ms.ones((ndim,)))
        self.bias = ms.Parameter(ms.zeros((ndim,))) if bias else None
        self.eps = 1e-5

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # 计算均值和方差（在最后一个维度上）
        mean = x.mean(-1, keep_dims=True)
        var = x.var(-1, keep_dims=True, unbiased=False)
        # 归一化
        x = (x - mean) / ops.sqrt(var + self.eps)
        # 应用缩放和偏移
        x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class CausalSelfAttention(nn.Cell):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # QKV投影（合并为一个线性层提高效率）
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropout层
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # 因果掩码（下三角矩阵）：防止关注未来token
        self.mask = np.tril(np.ones((config.block_size, config.block_size), dtype=ms.bool_))

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        B, T, C = x.shape  # 批次大小, 序列长度, 嵌入维度

        # QKV投影并拆分 (B, T, 3*C) -> (B, T, 3, H, C/H) -> 3*(B, H, T, C/H)
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.head_size)
        q, k, v = ops.unstack(qkv, axis=2)  # 拆分Q、K、V

        # 注意力计算: (B, H, T, T)
        attn = (q @ ops.transpose(k, (0, 1, 3, 2))) * (1.0 / ops.sqrt(ms.Tensor(self.head_size, dtype=ms.float32)))
        # 应用因果掩码（掩盖未来位置）
        attn = attn.masked_fill(self.mask[:T, :T] == 0, -1e18)
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # 注意力输出: (B, H, T, C/H) -> (B, T, H*C/H) = (B, T, C)
        y = attn @ v
        y = ops.transpose(y, (0, 2, 1, 3)).reshape(B, T, C)

        # 输出投影和残差dropout
        y = self.resid_drop(self.out_proj(y))
        return y


class MLP(nn.Cell):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Cell):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)  # 注意力前的层归一化
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)  # MLP前的层归一化
        self.mlp = MLP(config)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        # 注意力残差连接
        x = x + self.attn(self.ln1(x))
        # MLP残差连接
        x = x + self.mlp(self.ln2(x))
        return x


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
        assert config.vocab_size is not None, "vocab_size must be specified"
        assert config.block_size is not None, "block_size must be specified"
        self.config = config
        
         # 核心组件
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token嵌入
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # 位置嵌入
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.SequentialCell([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)  # 最终层归一化

        # 输出头（与Token嵌入共享权重，符合GPT-2设计）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.embedding_table = self.lm_head.weight  # 权重共享

        # 初始化参数
        self.apply(self._init_weights)

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
        B, T = idx.shape
        assert T <= self.config.block_size, f"序列长度{T}超过最大长度{self.config.block_size}"

        # 位置索引 (0, 1, ..., T-1)
        pos = ops.arange(0, T, dtype=ms.int64).reshape(1, T)

        # 嵌入计算：Token嵌入 + 位置嵌入
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # 经过所有Transformer块
        for block in self.blocks:
            x = block(x)

        # 最终层归一化和输出logits
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（如果提供了目标）
        loss = None
        if targets is not None:
            # 交叉熵损失：需要将logits展平为(B*T, vocab_size)，targets展平为(B*T,)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

        return logits, loss

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
        for _ in range(max_new_tokens):
            # 确保序列长度不超过block_size
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]

            # 获取预测logits
            logits, _ = self.construct(idx_cond)
            # 只关注最后一个token的预测
            logits = logits[:, -1, :] / temperature  # 应用温度缩放

            # Top-K过滤
            if top_k is not None:
                v, _ = ops.top_k(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))

            # 采样概率分布
            probs = ops.softmax(logits, axis=-1)
            idx_next = ops.multinomial(probs, num_samples=1)  # 采样下一个token

            # 拼接序列
            idx = ops.cat([idx, idx_next], axis=1)

        return idx



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
