"""
"""

from collections.abc import Callable

from torch import Tensor, inference_mode
from torch import tensor, ones, zeros, normal
from torch import int64
from torch import bincount

from torch.nn import Module

from .utils import Arg, empty_mps_cache
from .dataset import WaveDataset
from .model import multi_variate_gaussian


@inference_mode()
def recycle(
    model: Module,
    processor: Callable,
    train_ds: WaveDataset,
    **kwargs: Arg,
) -> bool:
    """
    """

    model.eval()

    batch_size = kwargs["eval_batch_size"]

    dim = model.vq.codebook_dim
    size = model.vq.codebook_size

    counts = zeros(size)
    for instance in map(processor, ds.iterate(batch_size)):
        inputs = instance["inputs"]
        attention_mask = instance["attention_mask"]

        batch_size = len(inputs)
        projected_inputs = model.feature_projector(inputs, attention_mask)
        logits = model.vq.logits(projected_inputs, attention_mask)
        codewords = logits.argmax(-1)
        counts += bincount(codewords.flatten(), minlength=size)

    mask = (counts == 0.0)
    if not mask.any():
        return False

    mask_length = int(mask.sum())
    tmp_vectors = multi_variate_gaussian(dim, mask_length)

    model.vq.codebook.codevectors[mask] = tmp_vectors

    if hasattr(model.vq.codebook, "precisions_sqrt"):
        model.vq.codebook.precisions_sqrt[mask] = ones(mask_length)

    empty_mps_cache()

    return True


def is_recyclable(**kwargs: Arg) -> bool:
    """
    """
    if kwargs["recycle"]:
        if kwargs["vq"] == "iso_gaussian":
            return True
        if kwargs["vq"] == "gaussian":
            return True
    return False


def main() -> None:
    """
    """

    pass

if __name__ == "__main__":
    main()
