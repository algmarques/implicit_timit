"""
"""

from torch import Tensor
from torch import dtype as DType
from torch import device as Device
from torch import int32
from torch import tensor, ones, ones_like, arange
from torch import cat
from torch import maximum

from torch import rand, multinomial


def get_ns_masked_spans(
    inputs_lengths: Tensor,
    masked_span_prob: float,
    masked_span_length: int,
    min_n_masked_spans: int = 1
) -> Tensor:
    """
    """

    min_n_masked_spans *= ones_like(inputs_lengths).to(inputs_lengths)

    ns_masked_spans = masked_span_prob * inputs_lengths
    ns_masked_spans = ns_masked_spans.ceil().to(inputs_lengths)

    ns_masked_spans = maximum(ns_masked_spans, min_n_masked_spans)

    tmp_mask = (ns_masked_spans * masked_span_length > inputs_lengths)
    ns_masked_spans[tmp_mask] = inputs_lengths[tmp_mask] // masked_span_length

    return ns_masked_spans


def get_masked_idxs(
    inputs: Tensor,
    masked_span_prob: float,
    masked_span_length: int,
    attention_mask: Tensor | None = None,
    min_n_masked_spans: int = 1,
    dtype: DType = int32,
    device: Device | str = "cpu"
) -> list[Tensor]:
    """
    """

    batch_size = len(inputs)

    inputs_lengths = ones_like(inputs[:, :, 0], dtype=dtype, device=device)
    if attention_mask is not None:
        inputs_lengths = attention_mask
    inputs_lengths = inputs_lengths.sum(-1)

    ns_masked_spans = get_ns_masked_spans(
        inputs_lengths,
        masked_span_prob,
        masked_span_length,
        min_n_masked_spans
    )

    ns_masked = masked_span_length * ns_masked_spans
    spans_ranges = inputs_lengths - ns_masked + 1

    masked_idxs = [None] * batch_size
    for i  in range(batch_size):
        n_masked_spans = ns_masked_spans[i]
        n_masked = ns_masked[i]
        spans_range = spans_ranges[i]

        weights = ones(spans_range).to(spans_range) / spans_range

        spans_idxs = multinomial(weights, n_masked_spans).to(n_masked_spans)
        sorted_spans_idxs, _ = spans_idxs.sort()
        sorted_spans_idxs += arange(0, n_masked, masked_span_length).to(
            n_masked
        )

        masked_idxs[i] = cat(
            [sorted_spans_idxs + i for i in range(masked_span_length)]
        )

    return masked_idxs


def get_negative_idxs(
    masked_idxs: Tensor,
    n_negatives: int
) -> list[Tensor]:
    """
    """

    batch_size = len(masked_idxs)
    negative_idxs = [None] * batch_size

    for i in range(batch_size):
        n_masked_idxs = len(masked_idxs[i])
        if n_masked_idxs < 2:
            raise ValueError
        weights = ones((n_masked_idxs, n_masked_idxs)) / (n_masked_idxs - 1)
        weights = weights.fill_diagonal_(0.0)
        tmp = multinomial(weights, n_negatives, replacement=True).to(
            masked_idxs[i]
        )
        negative_idxs[i] = masked_idxs[i][tmp]

    return negative_idxs


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
