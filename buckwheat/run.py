"""
Script for running the entire pipeline from the command line
"""
import argparse
import logging
import sys

from .language_recognition.utils import main as initialize_enry
from .main import tokenize_list_of_repositories
from .parsing.utils import main as initialize_parser


def main(args: argparse.Namespace) -> None:
    """
    :param args: arguments of parsing and modeling.
    :return: None.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    initialize_enry()
    initialize_parser()
    tokenize_list_of_repositories(repositories_file=args.input, output_dir=args.output,
                                  batch_size=int(args.batches), mode = args.parsing,
                                  gran=args.granularity, language=args.language, local=args.local,
                                  output_format=args.format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Full path to the input file. The list must contain the links to "
                             "GitHub in the default mode and paths to directories in the local "
                             "mode.")
    parser.add_argument("-o", "--output", required=True, help="Full path to the output directory.")
    parser.add_argument("-b", "--batches", default=10,
                        help="The size of the batch of projects that are saved to one file. "
                             "The default value is 10.")
    parser.add_argument("-p", "--parsing", choices=["counters", "sequences"], default="counters",
                        help="The mode of parsing. 'counters' returns Counter objects of subtokens"
                             " and their count, 'sequences' returns full sequences of subtokens "
                             "and their parameters: starting byte, starting line, starting symbol"
                             " in line. For the 'projects' granularity, only 'counters' are"
                             " available.")
    parser.add_argument("-g", "--granularity",
                        choices=["projects", "files", "classes", "functions"], default="files",
                        help="The granularity level of parsing: 'projects' for the level of "
                             "projects/directories, 'files' for the level of files, 'classes' for "
                             "the level of classes, and 'functions' for the level of functions.")
    parser.add_argument("-l", "--language", choices=["all", "JavaScript", "Python", "Java", "Go",
                                                     "C++", "Ruby", "TypeScript", "PHP", "C#", "C",
                                                     "Shell", "Rust", "Scala", "Swift", "Kotlin",
                                                     "Haskell"],
                        default="all", help="Language of parsing. By default, its all the "
                                            "languages supported in a given parsing granularity.")
    parser.add_argument("-f", "--format", choices=["wabbit", "json"], default="wabbit",
                        help="The output format for saving. 'wabbit' for Vowpal Wabbit, "
                             "'json' for JSON.")
    parser.add_argument("--local", action="store_true",
                        help="If passed, switches the tokenization into the local mode, where "
                             "the input list must contain paths to local directories.")
    args = parser.parse_args()
    main(args)
