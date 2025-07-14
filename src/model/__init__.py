"""
"""

from torch.nn import Module

from ..utils import Arg

from .wav2vec2 import Wav2Vec2
from .wav2vec2_e2e import Wav2Vec2E2E
from .wav2vec2_bert import Wav2Vec2Bert
from .wav2vec2_bert_e2e import Wav2Vec2BertE2E
from .wav2vec2_bert_ft import Wav2Vec2BertFT
from .random import multi_variate_gaussian


__all__ = [
    "codebooks",
    "convolutional",
    "encoder",
    "exceptions",
    "feed_forward",
    "internals",
    "layer_norm",
    "losses",
    "masking",
    "projector",
    "quantizers",
    "random",
    "self_attention",
    "wav2vec2_bert_e2e",
    "wav2vec2_bert_ft",
    "wav2vec2_bert",
    "wav2vec2_e2e",
    "wav2vec2"
]


def get_model(**kwargs: Arg) -> Module:
    """
    """

    if kwargs["zeta"]:
        if kwargs["ft"]:
            return Wav2Vec2BertFT(**kwargs)
        if kwargs["sigma"]:
            return Wav2Vec2BertE2E(**kwargs)
        return Wav2Vec2Bert(**kwargs)

    if kwargs["sigma"]:
        return Wav2Vec2E2E(**kwargs)
    return Wav2Vec2(**kwargs)
