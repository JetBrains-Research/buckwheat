"""
Script for running the entire pipeline from the command line
"""
import argparse

from .language_recognition.utils import main as initialize_enry
from .parsers.utils import main as initialize_parser
from .parsing import tokenize_list_of_repositories


def main(args: argparse.Namespace) -> None:
    """
    :param args: arguments of parsing and modeling.
    :return: None.
    """
    initialize_parser()
    initialize_enry()
    tokenize_list_of_repositories(repositories_file=args.input, output_dir=args.output,
                                  batch_size=int(args.batches), gran=args.granularity,
                                  local=args.local, output_format=args.format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Full path to the input file. The list must contain the links to "
                             "GitHub in the default mode and paths to directories in the local "
                             "mode.")
    parser.add_argument("-o", "--output", required=True, help="Full path to the output directory.")
    parser.add_argument("-b", "--batches", default=100,
                        help="The size of the batch of projects that are saved to one file.")
    parser.add_argument("-g", "--granularity",
                        choices=["projects", "files", "classes", "functions"], default="files",
                        help="The granularity level of parsing: 'projects' for the level of "
                             "projects/directories, 'files' for the level of files, 'classes' for "
                             "the level of classes, and 'functions' for the level of functions.")
    parser.add_argument("-l", "--local", action="store_true",
                        help="If passed, switches the tokenization into the local mode, where "
                             "the input list must contain paths to local directories.")
    parser.add_argument("-f", "--format", choices=["wabbit"], default="wabbit",
                        help="The output format for saving. 'wabbit' for Vowpal Wabbit. "
                             "More formats to come.")
    args = parser.parse_args()
    main(args)
