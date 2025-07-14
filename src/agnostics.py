"""
"""

from typing import Any
from collections.abc import Callable

from torch.nn import Module
from torch.optim import Optimizer

from .utils import Arg, empty_mps_cache
from .dataset import Dataset
from .metrics import get_metrics, aggregate
from .artifacts import get_artifacts


def train_epoch(
    model: Module,
    processor: Callable,
    optimizer: Optimizer,
    train_ds: Dataset,
    **kwargs: Arg
) -> None:
    """
    """

    model.train()

    batch_size = kwargs["train_batch_size"]
    dl = train_ds.iterate(batch_size, shuffle=True)
    for instance in map(processor, dl):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]
        target = instance["target"]

        optimizer.zero_grad()
        loss = model.loss(inputs, attention_mask, target)
        loss.backward()
        optimizer.step()
        empty_mps_cache()


def get(
    model: Module,
    processor: Callable,
    name: str,
    callable: Callable,
    ds_dict: dict[str, Dataset],
    **kwargs: Arg,
) -> dict[str, Any]:
    """
    """

    values = dict()
    for kind, ds in ds_dict.items():
        value = callable(model, processor, ds, **kwargs)
        values |= {kind + "_" + name: value}
        empty_mps_cache()

    return values


def evaluate(
    model: Module,
    processor: Callable,
    ds_dict: dict[str, Dataset],
    **kwargs: Arg
) -> dict[str, Any]:
    """
    """

    values = dict()
    metrics = get_metrics(**kwargs)

    for name, metric in metrics.items():
        values |= get(model, processor, name, metric, ds_dict, **kwargs)
    values = aggregate(values, **kwargs)

    return values


def compute(
    model: Module,
    processor: Callable,
    ds_dict: dict[str, Dataset],
    **kwargs: Arg
) -> dict[str, Any]:
    """
    """

    values = dict()
    artifacts = get_artifacts(**kwargs)

    for name, artifact in artifacts.items():
        values |= get(model, processor, name, artifact, ds_dict, **kwargs)

    return values


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
