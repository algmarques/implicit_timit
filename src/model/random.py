"""
"""

from torch import Tensor, zeros, eye
from torch.distributions import MultivariateNormal

def multi_variate_gaussian(dim: int, *shape: int) -> Tensor:
    """
    """

    dist = MultivariateNormal(zeros(dim), eye(dim))
    sample = dist.sample(shape)

    return sample


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()