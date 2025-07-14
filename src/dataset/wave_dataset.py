"""
"""

from __future__ import annotations

from pathlib import Path
from wave import open as open_wav
from array import array as Array

from ..utils import load_json
from .dataset import Dataset, Batch


class WaveDataset(Dataset):
    """
    """

    def __init__(self: WaveDataset, dir_pth: Path) -> None:
        """
        """

        super().__init__(dir_pth)


    @staticmethod
    def load_feature(inst_pth: Path) -> set[str]:
        """
        """

        feature = set()

        metadata_pth = inst_pth / "metadata.json"
        metadata = load_json(metadata_pth)

        return {"audio", "sample_rate"} | set(metadata)


    @staticmethod
    def load_instance(inst_pth: Path) -> Batch | None:
        """
        """

        audio_pth = inst_pth / "audio.wav"
        audio = load_wav(audio_pth)

        metadata_pth = inst_pth / "metadata.json"
        metadata = load_json(metadata_pth)

        return audio | metadata


def sample_size_to_typecode(sample_size: int) -> str | None:
    """
    """

    if sample_size == 1:
        return 'B'
    if sample_size == 2:
        return 'H'
    if sample_size == 4:
        return 'L'


def normalize(f: float, sample_size: int) -> float:
    """
    """

    #bound = ((256 ** sample_size) // 2 - 0.5)
    bound = (256 ** sample_size) // 2

    return (f - bound) / bound


def channel_to_fp_array(channel: bytes, sample_size: int) -> Array:
    """
    """

    typecode = sample_size_to_typecode(sample_size)
    arr = Array(typecode, channel)
    fp_arr = Array("d", arr)
    fp_arr = Array("d", map(lambda f: normalize(f, sample_size), fp_arr))

    return fp_arr


def load_wav(wav_pth: Path) -> dict[str, list[array] | int]:
    """
    """

    with open_wav(str(wav_pth), "rb") as wav_stream:

        n_channel = wav_stream.getnchannels()
        sample_size = wav_stream.getsampwidth()
        sample_rate = wav_stream.getframerate()
        n_frame = wav_stream.getnframes()

        buff = wav_stream.readframes(n_frame)

    audio = [None] * n_channel
    for i in range(n_channel):
        channel = buff[i::n_channel]
        audio[i] = channel_to_fp_array(channel, sample_size)

    return {"audio": audio, "sample_rate": sample_rate}


def main() -> None:
    """
    """

    pass


if __name__ == "__main__":
    main()
