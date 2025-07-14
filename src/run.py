"""
"""

from __future__ import annotations

from pathlib import Path
from json import dumps
from hashlib import shake_128
from itertools import product

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler

from .utils import Arg, filter_kwargs
from .utils import write_csv_header, write_csv_row, read_csv
from .utils import save_json
from .utils import save_torch, load_torch
from .utils import save_checkpoint, load_checkpoint
from .metrics import get_metrics


class Run(list):
    """
    """

    def __init__(self: Run, runs_dir_path: Path, **kwargs: Arg) -> Path:
        """
        """

        super().__init__()

        runs_dir_path.mkdir(exist_ok=True)

        for key, value in kwargs.items():
            setattr(self, key, value)

        tmp_kwargs = filter_kwargs(kwargs)
        self.id = self.get_id(tmp_kwargs)

        self.run_dir_path = runs_dir_path / self.id
        self.run_dir_path.mkdir(exist_ok=True)

        self.kwargs_file_path = self.run_dir_path / "kwargs.json"
        if not self.kwargs_file_path.exists():
            tmp_kwargs = kwargs | {"id": self.id}
            save_json(self.kwargs_file_path, tmp_kwargs)

        self.metrics_file_path = self.run_dir_path / "metrics.csv"

        self.metrics_names = list(get_metrics(**kwargs)) + ["loss"]
        self.header = product(["train", "dev", "test"], self.metrics_names)
        self.header = [kind + "_" + metric for kind, metric in self.header]

        if not self.metrics_file_path.exists():
            write_csv_header(self.metrics_file_path, self.header)
        for metrics in read_csv(self.metrics_file_path):
                metrics = {key: float(metrics[key]) for key in metrics}
                self.__iadd__([metrics])

        self.checkpoints_dir_path = self.run_dir_path / "checkpoints"
        self.checkpoints_dir_path.mkdir(exist_ok=True)
        self.latest_file_path = self.checkpoints_dir_path / "latest.pth"

        self.artifacts_dir_path = self.run_dir_path / "artifacts"
        self.artifacts_dir_path.mkdir(exist_ok=True)

    def get_id(self: Run, kwargs: dict[str, Arg]) -> str:
        """
        """

        tmp = dumps(kwargs, sort_keys=True).encode('utf-8')
        hash = shake_128(tmp, usedforsecurity=False)
        id = hash.hexdigest(6)

        return id

    @property
    def epoch(self: Run) -> int:
        """
        """

        return self.__len__()

    def finished(self: Run) -> int:
        """
        """

        return int(self.epoch > self.epochs)

    def __iter__(self: Run) -> int:
        """
        """

        while not self.finished():
            yield self.epoch

    @property
    def metrics(self: Run) -> dict[str, float]:
        """
        """

        return self[-1]

    def _checkpoint(
        self: Run,
        model: Module,
        optimizer: Optimizer,
        scheduler: Scheduler,
        metrics: dict[str, float]
    ) -> None:
        """
        """

        value = metrics["dev_loss"]
        history = [tmp["dev_loss"] for tmp in self[:-1]]

        if not history:
            return None

        file_path = self.checkpoints_dir_path / f"checkpoint.pth"
        if min(history) > value:
            save_checkpoint(file_path, model, optimizer, scheduler)

    def save(
        self: Run,
        model: Module,
        optimizer: Optimizer,
        scheduler: Scheduler,
        metrics: dict[str, float],
        artifacts: dict[str, Tensor],
    ) -> None:
        """
        """

        epoch = self.epoch
        self.__iadd__([metrics])
        write_csv_row(self.metrics_file_path, self.header, **metrics)
        save_checkpoint(self.latest_file_path, model, optimizer, scheduler)
        save_torch(self.artifacts_dir_path / f"{epoch}.pth", **artifacts)
        self._checkpoint(model, optimizer, scheduler, metrics)

    def load(
        self: Run,
        model: Module,
        optimizer: Optimizer,
        scheduler: Scheduler,
        metrics: dict[str, float],
        artifacts: dict[str, Tensor]
    ) -> bool:
        """
        """

        if self.latest_file_path.exists():
            load_checkpoint(self.latest_file_path, model, optimizer, scheduler)
            tmp_artifacts = load_torch(
                self.artifacts_dir_path / f"{self.epoch}.pth"
            )
            metrics.update(self.metrics)
            artifacts.update(tmp_artifacts)
            return True
        return False


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
