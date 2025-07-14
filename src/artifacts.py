"""
"""

from collections.abc import Callable

from torch import Tensor, inference_mode
from torch.nn import Module

from .utils import Arg
from .dataset import WaveDataset


@inference_mode()
def artifact(
    model: Module,
    processor: Callable,
    ds: WaveDataset,
    **kwargs: Arg,
) -> Tensor:
    """
    """

    model.eval()

    batch_size = kwargs["eval_batch_size"]

    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instances["attention_mask"]

    return None


def get_artifacts(**kwargs: Arg) -> dict[str, Callable]:
    """
    """

    artifacts = {
    }

    if kwargs["zeta"]:
        artifacts |= {}

    if kwargs["sigma"]:
        artifacts |= {}

    return artifacts


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
