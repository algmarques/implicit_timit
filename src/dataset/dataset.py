"""
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from pathlib import Path
from random import sample

from ..utils import load_json


type Resource = str | list[int] | Any
type Instance = dict[str, Resource]
type Batch = dict[str, list[Resource]]


class Dataset(list, ABC):
    """
    """


    def __init__(self: Dataset, dir_path: Path) -> None:
        """
        """

        self.dir_path = dir_path
        super().__init__()

        for inst_pth in self.dir_path.iterdir():
            if inst_pth.is_dir():
                super().append(inst_pth)

        super().sort()

        feature = set()
        for inst_pth in super().__iter__():
            feature |= self.load_feature(inst_pth)
        self.feature = feature


    @staticmethod
    @abstractmethod
    def load_feature(inst_pth: Path) -> set[str]:
        """
        """

        return None


    @staticmethod
    @abstractmethod
    def load_instance(inst_pth: Path) -> Batch | None:
        """
        """

        return None


    def extract(self: Dataset, instance_dir_path: list[Path]) -> Batch:
        """
        """

        instance = [self.load_instance(pth) for pth in instance_dir_path]

        batch = {feat: [None] * len(instance) for feat in self.feature}
        for i, inst in enumerate(instance):
            for feat in self.feature:
                batch[feat][i] = inst[feat]

        return batch


    def __getitem__(self: Dataset, key: int | slice) -> Batch:
        """
        """

        if isinstance(key, int):
            key = slice(key, key + 1, None)

        instance_dir_path = super().__getitem__(key)

        return self.extract(instance_dir_path)


    def iterate(
        self: Dataset,
        batch_size: int = 1,
        shuffle: bool = False
    ) -> Batch:
        """
        """

        size = super().__len__()

        tmp_instance_dir_path = list(super().__iter__())
        if shuffle:
            tmp_instance_dir_path = sample(tmp_instance_dir_path, k=size)

        n = size - size % batch_size
        for i in range(0, n, batch_size):
            j = i + batch_size
            batch = self.extract(tmp_instance_dir_path[i:j])
            yield batch


    def __iter__(self: Dataset) -> Batch:
        """
        """

        return self.iterate()


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
