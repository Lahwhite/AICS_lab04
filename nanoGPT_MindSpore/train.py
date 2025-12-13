import argparse

import mindspore as ms
import mindspore.communication.management as D
from mindspore import context
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.train.callback import (
    CheckpointConfig,
    LossMonitor,
    ModelCheckpoint,
    TimeMonitor,
)
from mindspore.train.model import Model

from src.dataset import create_dataset
from src.model import GPT, GPTConfig, GPTTrainOneStepCell, GPTWithLoss
from src.utils import LearningRate

VOCAB_SIZE = 5424


def run_train():
    """train function"""
    parser = argparse.ArgumentParser(description="Llama training")
    parser.add_argument(
        "--device_id", type=int, default=0, help="Device id, default is 0."
    )
    parser.add_argument(
        "--device_num", type=int, default=1, help="Use device nums, default is 1."
    )
    parser.add_argument(
        "--distribute",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Run distribute, default is false.",
    )
    parser.add_argument("--epoch", type=int, default=2, help="Epoches, default is 2.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset",
        help="Data path of your MindRecord files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="Batch size, default is 12.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=6e-5,
        help="Max learning rate, default is 6e-4.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=6e-6,
        help="Minimum learning rate, default is 6e-5.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=3000,
        help="Warmup iterations, default is 3000",
    )
    parser.add_argument(
        "--lr_decay_iters",
        type=int,
        default=750000,
        help="Learning rate decay iterations, default is 750000",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-1,
        help="Weight decay, default is 1e-1.",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm, default is 1.0.",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for Adam optimizer, default is 0.9.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="Beta2 for Adam optimizer, default is 0.95.",
    )
    parser.add_argument(
        "--sink_size",
        type=int,
        default=100,
        help="Sink size for every iteration, default is 100",
    )

    args_opt = parser.parse_args()

    device_id = args_opt.device_id
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    if args_opt.distribute == "true":
        D.init()
        device_num = args_opt.device_num
        rank = device_id % device_num
        print("device_id is {}, rank_id is {}".format(device_id, rank))

        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        rank = 0
        device_num = 1

    gpt_config = GPTConfig(vocab_size=(VOCAB_SIZE // 64 + 1) * 64)
    gpt = GPT(gpt_config)
    gpt_with_loss = GPTWithLoss(gpt)

    ds = create_dataset(
        batch_size=args_opt.batch_size,
        data_path=args_opt.data_path,
        device_num=device_num,
        rank=rank,
    )

    print("crated dataset")

    epoch_num = args_opt.epoch
    step_per_epoch = ds.get_dataset_size()

    print(f"step_per_epoch: {step_per_epoch}")

    lr = LearningRate(
        learning_rate=args_opt.learning_rate,
        min_lr=args_opt.min_lr,
        warmup_steps=args_opt.warmup_iters,
        decay_steps=args_opt.lr_decay_iters,
    )

    optimizer = gpt.configure_optimizers(
        weight_decay=args_opt.weight_decay,
        learning_rate=lr,
        betas=(args_opt.beta1, args_opt.beta2),
    )

    callback_size = args_opt.sink_size
    actual_epoch_num = int(epoch_num * step_per_epoch / callback_size)
    callback = [TimeMonitor(callback_size), LossMonitor(callback_size)]

    config_ck = CheckpointConfig(
        save_checkpoint_steps=step_per_epoch, keep_checkpoint_max=1
    )
    ckpoint_cb = ModelCheckpoint(prefix="GPT", config=config_ck)
    callback.append(ckpoint_cb)

    loss_scale_manager = ms.DynamicLossScaleManager(
        init_loss_scale=1024, scale_factor=2, scale_window=1000
    )

    model = GPTTrainOneStepCell(
        gpt_with_loss,
        optimizer,
        scale_sense=loss_scale_manager.get_update_cell(),
        gradient_clip=args_opt.grad_clip,
    )
    model = Model(model)
    print("start training")
    model.train(
        actual_epoch_num,
        ds,
        callbacks=callback,
        dataset_sink_mode=True,
        sink_size=callback_size,
    )


if __name__ == "__main__":
    set_seed(2025)
    run_train()
