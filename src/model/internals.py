"""
"""

from math import pi

from torch import Tensor
from torch import tensor, ones_like, zeros_like, rand
from torch import einsum, cdist

from torch.autograd import Function


def iso_gvq_logits(
    inputs: Tensor,
    codevectors: Tensor,
    precision_sqrt: Tensor
) -> Tensor:
    """
    """

    size, dim = list(codevectors.shape)

    precision = precision_sqrt ** 2
    const = dim * tensor(2 * pi).log() + 2 * tensor(size).log()
    dist = cdist(inputs, codevectors) * precision ** 2

    logits = - 1/2 * (dist - 2 * dim * precision.log() + const)

    return logits


def gvq_logits(
    inputs: Tensor,
    codevectors: Tensor,
    precisions_sqrt: Tensor
) -> Tensor:
    """
    """

    size, dim = list(codevectors.shape)

    precisions = precisions_sqrt ** 2
    const = dim * tensor(2 * pi).log() + 2 * tensor(size).log()
    dist = cdist(inputs, codevectors) * precisions ** 2

    logits = - 1/2 * (dist - 2 * dim * precisions.log() + const)

    return logits


class Choice(Function):
    """
    """

    @staticmethod
    # ctx type torch.autograd.function.GVQLogProbBackward defined dynamically
    # cannot annotate type
    def forward(
        ctx,
        probs: Tensor,
        codevectors: Tensor
    ) -> Tensor:
        """
        """

        ctx.save_for_backward(
            probs,
            codevectors
        )

        return codevectors[probs.argmax(dim=-1), :]

    @staticmethod
    def backward(ctx, choice_grad: Tensor) -> tuple[Tensor, ...]:
        """
        """

        probs, codevectors = ctx.saved_tensors

        probs_grad = einsum("...j, ij -> ...i", choice_grad, codevectors)
        codevectors_grad = einsum("...j, ...i -> ij", choice_grad, probs)

        return (probs_grad, codevectors_grad)


def choice(probs: Tensor, codevectors: Tensor) -> Tensor:
    """
    """

    return Choice.apply(probs, codevectors)


def guard(probs: Tensor, eps: float = 1e-6) -> Tensor:
    """
    """

    tmp = probs + eps

    return tmp / tmp.sum(-1, keepdim=True)


def norm(logits: Tensor) -> Tensor:
    """
    """

    tmp_logits = logits - logits.mean(-1, keepdim=True)
    norm_logits = tmp_logits.log_softmax(-1)

    return norm_logits


def gumbel_like(x: Tensor) -> Tensor:
    """
    """

    _mask = ones_like(x, dtype=bool, device="cpu")
    gumbel = ones_like(x, device="cpu")

    while _mask.any():
        gumbel[_mask] = rand(_mask.sum(), device="cpu")
        _mask = gumbel == 0.0

    return -(-gumbel.log()).log()


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    dim: int = -1
) -> Tensor:
    """
    """

    device = logits.device

    gumbels = gumbel_like(logits).to(device)
    soft = ((norm(logits) + gumbels) / tau).softmax(dim)

    if hard:
        idx = soft.max(dim, keepdim=True).indices
        hard = zeros_like(logits).scatter_(dim, idx, 1.0)
        return hard - soft.detach() + soft

    return soft


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
