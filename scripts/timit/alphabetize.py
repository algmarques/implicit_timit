"""
"""

import sys
from pathlib import Path

root_dir_path = Path(__file__).parent.parent.parent
sys.path += [str(root_dir_path)]


from argparse import ArgumentParser

from src.utils import save_json


def alphabetize_timit(source_dir_path: Path, target_dir_path: Path) -> None:
    """
    """

    target_dir_path.mkdir(parents=True, exist_ok=True)

    dictionary_file_path = source_dir_path / "TIMITDIC.TXT"
    dictionary = get_timit_dictionary(dictionary_file_path)
    save_json(target_dir_path / "dictionary.json", dictionary)

    phonemes = get_timit_phonemes()
    save_json(target_dir_path / "phonemes.json", phonemes)

    characters = get_timit_characters(dictionary)
    save_json(target_dir_path / "characters.json", characters)


def get_timit_dictionary(dictionary_file_path: Path) -> dict[str, int]:
    """
    """

    dictionary = ["#h", "#t"]

    with open(dictionary_file_path, 'r') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if line[0] == ";":
                continue
            word, _, _ = line.partition(' ')
            dictionary += [word]

    dictionary.sort()

    return {word: i for i, word in enumerate(dictionary)}


def get_timit_phonemes() -> dict[str, int]:
    """
    """

    phonemes = [
        # stops
        "b", "d", "g", "p", "t", "k", "dx", "q",

        # affricates
        "jh", "ch",

        # fricates
        "s", "sh", "z", "zh", "f", "th", "v", "dh",

        # nasals
        "m", "n", "ng", "em", "en", "eng", "nx",

        # semi-vowels and glides
        "l", "r", "w", "y", "hh", "hv", "el",

        # vowels
        "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
        "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h",

        # others
        "pau", "epi", "1", "2", "#h", "#t"
    ]

    phonemes.sort()

    return {phoneme: i for i, phoneme in enumerate(phonemes)}


def get_timit_characters(dictionary: dict[str, int]) -> dict[str, int]:
    """
    """

    characters = set([',', ';', "#h", "#t"])

    for word in dictionary:
        characters |= set(word)

    characters -= set(['.', '-', '_', '~', '#'])

    characters = list(characters)
    characters.sort()

    return {char: i for i, char in enumerate(characters)}


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

    alphabetize_timit(args.source_dir_path, args.target_dir_path)


if __name__ == "__main__":
    main()
