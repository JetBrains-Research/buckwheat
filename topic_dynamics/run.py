"""
Script for running the entire pipeline from the command line
"""
import argparse

from .language_recognition.utils import main as initialize_enry
from .parsers.utils import main as initialize_parser
from .parsing import slice_and_parse


def main(args: argparse.Namespace) -> None:
    """
    :param args: arguments of parsing and modeling.
    :return: None.
    """
    initialize_parser()
    initialize_enry()
    slice_and_parse(repositories_file=args.input, output_dir=args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Full path to the input file.")
    parser.add_argument("-o", "--output", required=True, help="Full path to the output directory.")
    args = parser.parse_args()
    main(args)
