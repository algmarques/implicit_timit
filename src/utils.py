"""
"""

from pathlib import Path
from typing import Any

from json import dump as _json_dump
from json import load as _json_load
from csv import DictReader, DictWriter

from torch import Tensor

from torch import save as _save_torch
from torch import load as _load_torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler

from torch.mps import empty_cache as _empty_mps_cache
from torch.backends.mps import is_available as is_mps_available


Arg = str | int | float


def save_checkpoint(
    file_path: Path,
    model: Module,
    optimizer: Optimizer,
    scheduler: Scheduler
) -> None:
    """
    """

    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }

    _save_torch(state, file_path)


def load_checkpoint(
    file_path: Path,
    model: Module,
    optimizer: Optimizer,
    scheduler: Scheduler
) -> None:
    """
    """

    run_state = _load_torch(file_path, weights_only=True)

    model_state = run_state["model_state"]
    optimizer_state = run_state["optimizer_state"]
    scheduler_state = run_state["scheduler_state"]

    if model:
        model.load_state_dict(model_state)

    if optimizer:
        optimizer.load_state_dict(optimizer_state)

    if scheduler:
        scheduler.load_state_dict(scheduler_state)


def save_json(file_path: Path, obj: Any, sort_keys=True) -> None:
    """
    """

    with open(file_path, "w") as stream:
        _json_dump(obj, stream, indent=4, sort_keys=sort_keys)


def load_json(file_path: Path) -> Any:
    """
    """

    with open(file_path, "r") as stream:
        obj = _json_load(stream)

    return obj


def save_torch(file_path: Path, **tensors: Tensor) -> None:
    """
    """

    if tensors:
        _save_torch(tensors, file_path)


def load_torch(file_path: Path) -> dict[str, Tensor]:
    """
    """

    if file_path.exists():
        return _load_torch(file_path)
    return dict()


def write_csv_header(csv_file_path: Path, header: list[str]) -> None:
    """
    """

    with open(csv_file_path, 'w') as csv_file:
        writer = DictWriter(csv_file, header)
        writer.writeheader()


def write_csv_row(csv_file_path: Path, header: list[str], **row: Any) -> None:
    """
    """

    if row:
        with open(csv_file_path, 'a') as csv_file:
            writer = DictWriter(csv_file, header)
            writer.writerow(row)


def read_csv(csv_file_path: Path) -> dict:
    """
    """

    with open(csv_file_path) as csv_file:
        for row in DictReader(csv_file):
            yield row


def union(some: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """
    """

    for key, value in other.items():
        if key in some and some[key] is None:
            some[key] = other[key]
        if key not in some:
            some[key] = other[key]

    return some


def filter_kwargs(some: dict[str, Arg]) -> dict[str, Arg]:
    """
    """

    other = dict()
    for key in some:
        if not key in {"epochs", "eval_batch_size", "skip"}:
            other[key] = some[key]

    return other


def process_kwargs(dir_path: Path, **kwargs: Arg) -> dict[str, Arg]:
    """
    """

    target = kwargs["target"]

    dct_file_path = dir_path / f"{target}.json"
    dct = load_json(dct_file_path)

    return kwargs | {"n_classes": len(dct)}


def get_default_kwargs(dir_path: Path) -> dict[str, Arg]:
    """
    """

    default_kwargs_file_path = dir_path / "default_kwargs.json"

    return load_json(default_kwargs_file_path)


def empty_mps_cache() -> None:
    """
    # explain memory leak
    """

    if is_mps_available():
        _empty_mps_cache()


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
