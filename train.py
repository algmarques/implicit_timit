"""
"""

from pathlib import Path
from sys import argv, executable, exit
from subprocess import run


def run_epoch() -> bool:
    """
    """

    root_dir_path = Path(__file__).parent
    file_path = str(root_dir_path / "scripts" / "fork.py")
    args = f"{executable} -B {file_path} " + " ".join(argv[1:])

    process = run(args, capture_output=True, shell=True)

    if process.stdout == b'1\n':
        return not True
    if process.stdout == b'0\n':
        return not False

    print(process.stderr.decode('utf-8'))
    exit()


def main() -> None:
    """
    """

    while run_epoch():
        pass


if __name__ == "__main__":
    main()
