"""
"""

import sys
from pathlib import Path

root_dir_path = Path(__file__).parent.parent.parent
sys.path += [str(root_dir_path)]


from typing import Any
from collections.abc import Hashable, Iterable

from argparse import ArgumentParser

from src.utils import save_json, load_json


def dict_map(
    some_dict: dict[Hashable, Any],
    iterable: Iterable[Hashable]
) -> list[Any]:
    """
    """

    tmp = []
    for key in iterable:
        if key in some_dict:
            tmp += [some_dict[key]]

    return tmp


def process_timit(source_dir_path: Path, target_dir_path: Path) -> None:
    """
    """

    target_dir_path.mkdir(parents=True, exist_ok=True)

    for audio_file_path in source_dir_path.glob("**/*.WAV.wav"):

        segments = audio_file_path.parts
        split = segments[2].lower()
        dialect_id = segments[3].lower()
        speaker_id = segments[4].lower()
        sentence_id, _, _ = segments[5].lower().partition('.')

        instance_stem = f"{dialect_id}_{speaker_id}_{sentence_id}"
        instance_dir_path = target_dir_path / split / instance_stem

        orig_dir_path = instance_dir_path / ".orig"
        orig_dir_path.mkdir(parents=True, exist_ok=True)

        audio_file_path.replace(orig_dir_path / "audio.wav")

        audio_file_path = audio_file_path.with_suffix('')

        phonemes_file_path = audio_file_path.with_suffix(".PHN")
        phonemes_file_path.replace(orig_dir_path / "phonemes.txt")

        words_file_path = audio_file_path.with_suffix(".WRD")
        words_file_path.replace(orig_dir_path / "words.txt")

        sentence_file_path = audio_file_path.with_suffix(".TXT")
        sentence_file_path.replace(orig_dir_path / "sentence.txt")

    for instance_dir_path in target_dir_path.glob("*/*/"):
        process_timit_instance(target_dir_path, instance_dir_path)


def process_timit_instance(
    target_dir_path: Path,
    instance_dir_path: Path
) -> None:
    """
    """

    orig_dir_path = instance_dir_path / ".orig"
    metadata_file_path = instance_dir_path / "metadata.json"

    dialect_id, speaker_id, sentence_id = instance_dir_path.stem.split("_")

    audio_file_path = instance_dir_path / "audio.wav"
    audio_buff = (orig_dir_path / "audio.wav").read_bytes()
    audio_file_path.write_bytes(audio_buff)

    words_dict = load_json(target_dir_path / "dictionary.json")
    phonemes_dict = load_json(target_dir_path / "phonemes.json")
    characters_dict = load_json(target_dir_path / "characters.json")

    sentence = read_timit_tokens(orig_dir_path / "sentence.txt")
    words = read_timit_tokens(orig_dir_path / "words.txt")
    phonemes = read_timit_tokens(orig_dir_path / "phonemes.txt")
    characters = read_characters(characters_dict, words)

    sentence = sentence[1]
    characters = dict_map(characters_dict, characters)
    words = dict_map(words_dict, words)
    phonemes = dict_map(phonemes_dict, phonemes)

    metadata = {
        "dialect_id": dialect_id,
        "speaker_id": speaker_id,
        "sentence_id": sentence_id,
        "sentence": sentence,
        "words": words,
        "phonemes": phonemes,
        "characters": characters
    }

    save_json(metadata_file_path, metadata, sort_keys=False)


def read_characters(
    characters_dict: dict[str, int],
    words: list[str]
) -> list[int]:
    """
    """

    words = words[1: -1]

    characters = ["#h"]

    for word in words:
        for char in word:
            if char in characters_dict:
                characters += [char]
    characters += ["#t"]

    return characters


def read_timit_tokens(file_path: Path) -> list[str]:
    """
    """

    tokens = ["#h"]
    with open(file_path, "r") as stream:
        while True:
            token = stream.readline()
            if not token:
                break
            _, _, token = token[:-1].split(' ', 2)
            if token == "h#":
                continue
            tokens += [token]
    tokens += ["#t"]

    return tokens


def main() -> None:
    """
    """

    parser = ArgumentParser(prog="TIMIT processor")
    parser.add_argument(
        "source_dir_path",
        type=Path,
        help="path of the original extracted dir"
    )
    parser.add_argument(
        "target_dir_path",
        type=Path,
        help="path of the dataset target dir"
    )

    args = parser.parse_args()

    process_timit(args.source_dir_path, args.target_dir_path)


if __name__ == "__main__":
    main()
