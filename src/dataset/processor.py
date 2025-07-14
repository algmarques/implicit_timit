"""
"""

from collections.abc import Callable
from functools import partial

from transformers import SeamlessM4TFeatureExtractor as FeatureExtractor

from torch import Tensor
from torch import tensor
from torch import int64
from torch import get_default_device
from torch import get_default_dtype

from ..utils import Arg
from .dataset import Instance, Batch


_feature_extractor = FeatureExtractor()


def pad(target: list[list[int]]) -> Tensor:

    max_sequence_length = max(map(len, target))
    for i, trgt in enumerate(target):
        target[i] += [1] * (max_sequence_length - len(trgt) + 1)

    target = tensor(target).to(int64)

    return target


def processor(batch: Batch, **kwargs: Arg) -> Batch:
    """
    """

    # SeamlessM4TFeatureExtractor used old-school tensor initializer
    # must pass default device and dtype
    device = get_default_device()
    dtype = get_default_dtype()

    max_length = kwargs["length"] * 2

    extracted = _feature_extractor(
        batch["audio"],
        return_tensors="pt",
        sampling_rate=16_000,
        padding="max_length",
        max_length=max_length,
        truncation=True
    )

    inputs = extracted["input_features"].to(device, dtype)
    attention_mask = extracted["attention_mask"].to(device, bool)

    target = batch[kwargs["target"]]
    target = pad(target)

    return {
        "inputs": inputs,
        "attention_mask": attention_mask,
        "target": target,
    }


def get_processor(**kwargs: Arg) -> Callable:
    """
    """

    return partial(processor, **kwargs)


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
