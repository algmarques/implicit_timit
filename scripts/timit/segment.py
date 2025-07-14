"""
"""

from pathlib import Path
from typing import Any
from random import seed as set_seed
from math import floor
from random import sample as get_sample
from csv import DictReader, DictWriter
from argparse import ArgumentParser


def segment(
    source_dir_path: Path,
    target_dir_path: Path,
    fraction: float,
    seed: int
) -> None:
    """
    """

    target_dir_path.mkdir(parents=True, exist_ok=True)

    instance = list(source_dir_path.iterdir())
    n = len(instance)

    set_seed(seed)
    sample_size = floor(fraction * n)
    sample = get_sample(range(n), sample_size)

    for i in sample:
        inst = instance[i]
        name = inst.stem
        s_dir_pth = source_dir_path / name
        t_dir_pth = target_dir_path / name
        s_dir_pth.rename(t_dir_pth)


def main() -> None:
    """
    """

    parser = ArgumentParser(prog="TIMIT partitioner")
    parser.add_argument(
        "source_dir_path",
        type=Path,
        help=""
    )
    parser.add_argument(
        "target_dir_path",
        type=Path,
        help=""
    )
    parser.add_argument(
        "--fraction",
        "-f",
        type=float,
        default=1 / 3,
        help=""
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=""
    )
    args = parser.parse_args()

    source_dir_path = args.source_dir_path
    target_dir_path = args.target_dir_path
    fraction = args.fraction
    seed = args.seed

    segment(source_dir_path, target_dir_path, fraction, seed)


if __name__ == "__main__":
    main()
