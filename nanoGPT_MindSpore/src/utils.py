import os
from typing import List

import mindspore as ms
import mindspore.numpy as np
from mindspore.nn.learning_rate_schedule import (
    CosineDecayLR,
    LearningRateSchedule,
    WarmUpLR,
)
from mindspore.ops import functional as F

SEQ_LEN = 256


def read_jinyong(path: str) -> str:
    """read Jin Yong fictions"""
    files: List[str] = []
    if os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                # 只读取 .txt 文件，跳过 .bin、.json 等其他文件
                if filename.endswith(".txt"):
                    files.append(os.path.join(root, filename))
    elif os.path.isfile(path):
        files.append(path)
    else:
        raise ValueError("Invalid path")

    data = ""
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data += f.read()
        except UnicodeDecodeError:
            # 如果 UTF-8 失败，尝试用 GBK 编码（中文常用）
            with open(file_path, "r", encoding="gbk", errors="ignore") as f:
                data += f.read()

    return data


class LearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for GPT network.
    """

    def __init__(
        self,
        learning_rate,
        min_lr,
        warmup_steps,
        decay_steps,
    ):
        super(LearningRate, self).__init__()

        self.warmup_flag = True
        self.warmup_lr = WarmUpLR(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
        )

        self.consine_decay_lr = CosineDecayLR(
            min_lr=min_lr, max_lr=learning_rate, decay_steps=decay_steps
        )

        self.warmup_steps = np.array([warmup_steps]).astype(np.float32)

        self.one = np.array([1.0]).astype(np.float32)

    def construct(self, global_step: ms.Tensor):
        """dynamic learning rate"""
        decay_lr = self.consine_decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = F.greater(self.warmup_steps, global_step).float()
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr
