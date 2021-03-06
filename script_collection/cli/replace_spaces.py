#! /usr/bin/python
import os
import argparse
from shutil import copy as copy_file


_UNICODE_SPACES = [  # http://jkorpela.fi/chars/spaces.html
    "\u0020",
    "\u00A0",
    "\u1680",
    "\u180E",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200A",
    "\u200B",
    "\u202F",
    "\u205F",
    "\u3000",
    "\uFEFF",
]


def _main():
    """ Entrypoint """
    parser = argparse.ArgumentParser(description="replace_spaces")
    parser.add_argument("file_path", nargs="+", help="File(s) to handle.")
    parser.add_argument("--char", type=str, default=".",
                        help="Character used to replace spaces. (default '.')")
    parser.add_argument("-c", action="store_true",
                        help="If given copies the file(s) with the new name.")
    parser.add_argument("-n", action="store_true",
                        help="Dry run, only prints the new name.")
    args = parser.parse_args()
    file_paths = args.file_path
    replace_char = args.char
    copy = args.c
    dry = args.n
    if dry:
        print("Dry mode, would {} modified files like".format(
            "copy" if copy else "rename"))

    for file_path in file_paths:
        base = os.path.basename(file_path)
        path = os.path.dirname(file_path)

        for space in _UNICODE_SPACES:
            base = base.replace(space, replace_char)

        out_path = os.path.join(path, base)

        if copy and not dry:
            copy_file(file_path, out_path)
        elif not dry:
            os.rename(file_path, out_path)
        else:
            print("- {}".format(out_path))


if __name__ == "__main__":
    _main()
