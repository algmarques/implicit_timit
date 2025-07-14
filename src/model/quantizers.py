"""
"""

from __future__ import annotations

from torch import Tensor
from torch import zeros
from torch import tensor_split, cat, stack

from torch.nn import Module
from torch.nn.functional import one_hot

from ..utils import Arg
from .internals import choice, gumbel_softmax, norm
from .codebooks import IsoGaussianCodeBook, GaussianCodeBook
from .codebooks import LinearCodeBook


class _VectorQuantizer(Module):
    """
    """

    _required = {
        "n_groups",
        "codebook_size",
        "codebook_dim",
        "freeze",
        "gumbel_softmax"
    }

    def __init__(self: _VectorQuantizer, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.tau = 1.0

    def logits(
        self: _VectorQuantizer,
        inputs: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        batch_size = len(inputs)
        batch_shape = list(inputs.shape)
        hidden_dim = batch_shape.pop()
        sequence_length = batch_shape.pop()

        size = self.codebook_size

        split_inputs = tensor_split(
            inputs[attention_mask],
            self.n_groups,
            dim=-1
        )

        logits = zeros(batch_size, sequence_length, self.n_groups, size)
        for i, inputs in enumerate(split_inputs):
            logits[:, :, i, :][attention_mask] = self.codebook(inputs)

        return logits

    def forward(
        self: _VectorQuantizer,
        inputs: Tensor,
        attention_mask: Tensor
    ) -> Tensor:
        """
        """

        batch_size = len(inputs)
        batch_shape = list(inputs.shape)
        hidden_dim = batch_shape.pop()
        sequence_length = batch_shape.pop()

        split_inputs = tensor_split(
            inputs[attention_mask],
            self.n_groups,
            dim=-1
        )

        split_codevectors = [None] * self.n_groups
        split_codewords = [None] * self.n_groups
        for i, inputs in enumerate(split_inputs):
            logits = self.codebook(inputs)
            codewords = logits.argmax(-1)
            if self.gumbel_softmax:
                probs = gumbel_softmax(logits, tau=self.tau)
            else:
                probs = norm(logits).exp()

            if not self.training:
                probs = one_hot(codewords, self.codebook_size).to(probs)

            codevectors = choice(probs, self.codebook.codevectors)

            split_codevectors[i] = codevectors
            split_codewords[i] = codewords

        tmp_codevectors = cat(split_codevectors, dim=-1)
        tmp_codewords = stack(split_codewords, dim=-1)

        codevectors = zeros(batch_size, sequence_length, hidden_dim)
        codewords = zeros(batch_size, sequence_length, self.n_groups)
        codewords = codewords.to(tmp_codewords)

        codevectors[attention_mask] = tmp_codevectors
        codewords[attention_mask] = tmp_codewords

        return codevectors, codewords.detach()


class IsoGaussianVectorQuantizer(_VectorQuantizer):
    """
    """

    def __init__(self: IsoGaussianVectorQuantizer, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)
        self.codebook = IsoGaussianCodeBook(**kwargs)


class GaussianVectorQuantizer(_VectorQuantizer):
    """
    """

    def __init__(self: GaussianVectorQuantizer, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)
        self.codebook = GaussianCodeBook(**kwargs)


class LinearVectorQuantizer(_VectorQuantizer):
    """
    """

    def __init__(self: LinearVectorQuantizer, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)
        self.codebook = LinearCodeBook(**kwargs)


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
