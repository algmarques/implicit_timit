"""
"""

import sys
from pathlib import Path

root_dir_path = Path(__file__).parent.parent
sys.path += [str(root_dir_path)]


from argparse import ArgumentParser
from pathlib import Path
from random import seed

from torch.autograd import set_detect_anomaly
from torch import manual_seed

from src.fork import fork
from src.utils import union, get_default_kwargs


def main() -> None:
    """
    """

    parser = ArgumentParser(
        description=""
    )
    parser.add_argument(
        "vq",
        type=str,
        help=""
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help=""
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help=""
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help=""
    )
    parser.add_argument(
        "--beta",
        type=float,
        help=""
    )
    parser.add_argument(
        "--zeta",
        type=float,
        help=""
    )
    parser.add_argument(
        "--sigma",
        type=float,
        help=""
    )
    parser.add_argument(
        "--detach",
        "-d",
        type=int,
        help=""
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help=""
    )
    parser.add_argument(
        "--ft",
        type=int,
        help=""
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help=""
    )

    args = parser.parse_args()
    kwargs = vars(args).copy()
    kwargs.pop("seed")

    root_dir_path = Path(__file__).parent.parent
    data_dir_path = root_dir_path / "data"
    runs_dir_path = root_dir_path / "runs"

    default_kwargs = get_default_kwargs(root_dir_path)
    kwargs = union(kwargs, default_kwargs)

    manual_seed(args.seed)
    seed(args.seed)

    # set_detect_anomaly(True)

    fork(runs_dir_path, data_dir_path, **kwargs)


if __name__ == "__main__":
    main()
