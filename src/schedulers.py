"""
"""

from __future__ import annotations

from math import sqrt, exp
from functools import partial

from torch import get_default_device, set_default_device

from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler as Scheduler
from torch.optim.lr_scheduler import LambdaLR

from .utils import Arg


def sqrt_decay(epoch: int, warmup_epochs: int = 0) -> float:
    """
    """

    if epoch > warmup_epochs:
        return 1 / sqrt(epoch - warmup_epochs)
    return (epoch + 1) / (warmup_epochs + 1)

def exp_decay(epoch: int, warmup_epochs: int = 0) -> float:
    """
    """

    if epoch > warmup_epochs:
        return exp(warmup_epochs - epoch)
    return 1.0


def get_optimizer(model: Module, **kwargs: Any) -> Optimizer:
    """
    """
    # jez! another hack arround mps nasty bugs!

    device = get_default_device()
    set_default_device("cpu")

    optimizer = Adam(
        model.parameters(),
        lr=kwargs["lr"],
        betas=kwargs["betas"],
        eps=1e-6
    )

    set_default_device(device)

    return optimizer


def get_scheduler(optimizer: Optimizer, **kwargs: Arg) -> Scheduler:
    """
    """

    _sqrt_decay = partial(sqrt_decay, warmup_epochs=kwargs["warmup_epochs"])

    return LambdaLR(optimizer, _sqrt_decay)


class Annealer(object):
    """
    """

    _required = {
        "tau_ub",
        "tau_lb"
    }

    def __init__(self: Annealer, model: Module, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.tau = model.vq.tau

    def step(self: Annealer, epoch: int) -> None:
        """
        """

        self.tau = sqrt_decay(epoch) * self.tau_ub + self.tau_lb


class Teacher(object):
    """
    """

    _required = {
        "omega_ub",
        "omega_lb",
        "teacher_forcing"
    }

    def __init__(self: Teacher, model: Module, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.omega = self.omega_ub
        if hasattr(model, "omega"):
            self.omega = model.omega

    def step(self: Teacher, epoch: int) -> None:
        """
        """

        omega = sqrt_decay(epoch) * self.omega_ub + self.omega_lb
        self.omega = self.teacher_forcing * omega


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
