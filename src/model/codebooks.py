"""
"""

from __future__ import annotations

from typing import Never
from abc import ABC, abstractmethod

from torch import Tensor, tensor, ones
from torch.nn import Module, Parameter, Linear

from ..utils import Arg
from .random import multi_variate_gaussian
from .internals import iso_gvq_logits, gvq_logits


class _CodeBook(Module, ABC):
    """
    """

    _required = {
        "codebook_size",
        "codebook_dim",
        "freeze"
    }

    def _init_codevectors(self: _CodeBook) -> Parameter:
        """
        """

        dim = self.codebook_dim
        size = self.codebook_size
        requires_grad = not self.freeze

        codevectors = multi_variate_gaussian(dim, size)

        return Parameter(codevectors, requires_grad=requires_grad)

    def __init__(self: _CodeBook, **kwargs: Arg) -> None:
        """
        """

        super().__init__()

        for attr in self._required:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                raise ValueError()

        self.codevectors = self._init_codevectors()

    @abstractmethod
    def forward(self: _CodeBook) -> Tensor:
        """
        """

        raise NotImplementedError()

    def _guard(self: _CodeBook) -> Never:
        """
        """

        if self.codewords.isnan().any():
            raise ValueError()


class IsoGaussianCodeBook(_CodeBook):
    """
    """

    @staticmethod
    def _init_precision_sqrt() -> Parameter:
        """
        """

        precision_sqrt = tensor([1.0]).sqrt()

        return Parameter(precision_sqrt, requires_grad=True)

    def __init__(self: IsoGaussianCodeBook, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)

        self.precision_sqrt = self._init_precision_sqrt()

    def forward(self: IsoGaussianCodeBook, inputs: Tensor) -> Tensor:
        """
        """

        logits = iso_gvq_logits(inputs, self.codevectors, self.precision_sqrt)

        return logits

    def _guard(self: IsoGaussianCodeBook) -> Never:
        """
        """

        super()._guard()

        if self.precision_sqrt.isnan().any():
            raise ValueError()


class GaussianCodeBook(_CodeBook):
    """
    """

    def _init_precisions_sqrt(self: GaussianCodeBook) -> Parameter:
        """
        """

        return Parameter(ones(self.codebook_size), requires_grad=True)

    def __init__(self: GaussianCodeBook, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)

        self.precisions_sqrt = self._init_precisions_sqrt()

    def forward(self: GaussianCodeBook, inputs: Tensor) -> Tensor:
        """
        """

        logits = gvq_logits(inputs, self.codevectors, self.precisions_sqrt)

        return logits

    def _guard(self: GaussianCodeBook) -> Never:
        """
        """

        super()._guard()

        if self.precisions_sqrt.isnan().any():
            raise ValueError()


class LinearCodeBook(_CodeBook):
    """
    """

    def __init__(self: LinearCodeBook, **kwargs: Arg) -> None:
        """
        """

        super().__init__(**kwargs)

        self.linear = Linear(self.codebook_dim, self.codebook_size)

    def forward(self: LinearCodeBook, inputs: Tensor) -> Tensor:
        """
        """

        logits = self.linear(inputs)

        return logits


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
