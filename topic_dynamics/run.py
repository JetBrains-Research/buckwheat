"""
Script for running the entire pipeline from the command line
"""
import argparse

from .language_recognition.utils import main as initialize_enry
from .parsers.utils import main as initialize_parser
from .parsing import tokenize_repositories


def main(args: argparse.Namespace) -> None:
    """
    :param args: arguments of parsing and modeling.
    :return: None.
    """
    initialize_parser()
    initialize_enry()
    tokenize_repositories(repositories_file=args.input, output_dir=args.output,
                          batch_size=int(args.batches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Full path to the input file.")
    parser.add_argument("-o", "--output", required=True, help="Full path to the output directory.")
    parser.add_argument("-b", "--batches", default=100, help="The size of the batch of projects.")
    args = parser.parse_args()
    main(args)
