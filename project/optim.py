import torch.nn as nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Tuple


def build_sgd_optimizer(model: nn.Module, lr: float, weight_decay: float) -> Optimizer:
    return SGD(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_adamw_warmup_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> Optimizer:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_lr_scheduler(optimizer: Optimizer, warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
