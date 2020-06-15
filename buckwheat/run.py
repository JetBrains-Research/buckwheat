"""
Script for running the entire pipeline from the command line
"""
import argparse
import logging
import sys

from .language_recognition.utils import main as initialize_enry
from .parsing.utils import main as initialize_parser
from .tokenizer import tokenize_list_of_repositories
from .utils import PARSING_MODES, GRANULARITIES, OUTPUT_FORMATS


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
                                  batch_size=int(args.batches), mode=args.parsing,
                                  gran=args.granularity, languages=args.languages, local=args.local,
                                  output_format=args.format,
                                  identifiers_verbose=args.identifiers_verbose,
                                  subtokenize=args.subtokenize)


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
    parser.add_argument("-p", "--parsing", choices=PARSING_MODES,
                        default="sequences",
                        help="The mode of parsing. 'counters' returns Counter objects of subtokens"
                             " and their count, 'sequences' returns full sequences of subtokens "
                             "and their parameters: starting byte, starting line, starting symbol"
                             " in line. For the 'projects' granularity, only 'counters' are"
                             " available.")
    parser.add_argument("-g", "--granularity",
                        choices=GRANULARITIES, default="files",
                        help="The granularity level of parsing: 'projects' for the level of "
                             "projects/directories, 'files' for the level of files, 'classes' for "
                             "the level of classes, and 'functions' for the level of functions.")
    parser.add_argument("-l", "--languages", nargs="*",
                        help="Languages of parsing. By default, its all the "
                             "languages supported in a given parsing granularity.")
    parser.add_argument("-f", "--format", choices=OUTPUT_FORMATS,
                        default="wabbit", help="The output format for saving. "
                                               "'wabbit' for Vowpal Wabbit, 'json' for JSON.")
    parser.add_argument("-v", "--identifiers_verbose", action="store_true",
                        help="If passed, all the identifiers will be saved with their coordinates "
                             "(starting byte, starting line, starting column). Doesn't work for "
                             "the 'counters' mode.")
    parser.add_argument("-s", "--subtokenize", action="store_true",
                        help="If passed, all the tokens will be split into subtokens by "
                             "camelCase and snake_case, and also stemmed. For the details of "
                             "subtokenization, see subtokenizer.py.")
    parser.add_argument("--local", action="store_true",
                        help="If passed, switches the tokenization into the local mode, where "
                             "the input list must contain paths to local directories.")
    args = parser.parse_args()
    main(args)
