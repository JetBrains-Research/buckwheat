"""
Script for running the entire pipeline from the command line
"""
import argparse
import datetime
import os

from .language_recognition.utils import main as initialize_enry
from .modeling import model_topics
from .parsers.utils import main as initialize_parser
from .parsing import slice_and_parse_full_files, slice_and_parse_diffs


def main(args: argparse.Namespace) -> None:
    """
    :param args: arguments of parsing and modeling.
    :return: None.
    """
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    with open(os.path.abspath(os.path.join(args.output, "log.txt")), "w+") as fout:
        fout.write("Topic modeling pipeline launched: {datetime}\n"
                   "\n"
                   "--Slicing parameters--\n"
                   "Starting date: {start_date}\n"
                   "Number of temporal slices: {slices}\n"
                   "Difference between slices in days: {days}\n"
                   "\n"
                   "--Parsing parameters--\n"
                   "A list of input projects: {input}\n"
                   "An output folder: {output}\n"
                   "Parsing mode: {mode}\n"
                   "\n"
                   "--Modeling parameters--\n"
                   "Number of topics: {topics}\n"
                   "Sparse Theta parameter: {sparse_theta}\n"
                   "Sparse Phi parameter: {sparse_phi}\n"
                   "Decorrelator Phi parameter: {decorrelator_phi}\n"
                   "Number of document passes: {document_passes}\n"
                   "Number of collection passes: {collection_passes}\n"
                   "\n"
                   "--Statistics parameters--\n"
                   "Number of topical files to save: {topical_files}"
                   .format(datetime=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                           start_date=args.start_date, slices=args.slices, days=args.days,
                           input=args.input, output=args.output, mode=args.mode,
                           topics=args.topics, sparse_theta=args.sparse_theta,
                           sparse_phi=args.sparse_phi, decorrelator_phi=args.decorrelator_phi,
                           document_passes=args.document_passes,
                           collection_passes=args.collection_passes,
                           topical_files=args.topical_files))
    with open(os.path.abspath(os.path.join(args.output, "log.txt"))) as fin:
        print(fin.read())
    initialize_parser()
    initialize_enry()
    if args.mode == "diffs":
        slice_and_parse_diffs(repository=args.input, output_dir=args.output,
                              n_dates=int(args.slices), day_delta=int(args.days),
                              start_date=args.start_date)
        model_topics(output_dir=args.output, n_topics=int(args.topics),
                     sparse_theta=float(args.sparse_theta), sparse_phi=float(args.sparse_phi),
                     decorrelator_phi=float(args.decorrelator_phi),
                     n_doc_iter=int(args.document_passes), n_col_iter=int(args.collection_passes),
                     n_files=int(args.topical_files), diffs=True)
    elif args.mode == "files":
        slice_and_parse_full_files(repository=args.input, output_dir=args.output,
                                   n_dates=int(args.slices), day_delta=int(args.days),
                                   start_date=args.start_date)
        model_topics(output_dir=args.output,  n_topics=int(args.topics),
                     sparse_theta=float(args.sparse_theta), sparse_phi=float(args.sparse_phi),
                     decorrelator_phi=float(args.decorrelator_phi),
                     n_doc_iter=int(args.document_passes), n_col_iter=int(args.collection_passes),
                     n_files=int(args.topical_files), diffs=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", choices=["diffs", "files"], default="diffs",
                        help="The mode of parsing: 'files' for tokenizing full files,"
                             " 'diffs' for only diffs of files (default).")
    parser.add_argument("-i", "--input", required=True, help="Full path to the input repository.")
    parser.add_argument("-o", "--output", required=True, help="Full path to the output directory.")
    parser.add_argument("-s", "--slices", default=24,
                        help="Number of temporal slices. Default number is 24.")
    parser.add_argument("-d", "--days", default=60,
                        help="Difference between slices in days. Default number is 60.")
    parser.add_argument("-start", "--start_date", default=None,
                        help="The starting (latest) date of the slicing, in the format YYYY-MM-DD,"
                             " the default value is the moment of calling.")
    parser.add_argument("-t", "--topics", default=45,
                        help="Number of topics in model. Default number is 45.")
    parser.add_argument("-st", "--sparse_theta", default=-0.15,
                        help="Sparse Theta parameter value. Default value is -0.15.")
    parser.add_argument("-sp", "--sparse_phi", default=-0.1,
                        help="Sparse Phi parameter value. Default value is -0.1.")
    parser.add_argument("-dp", "--decorrelator_phi", default=1.5e+5,
                        help="Decorrelator Phi parameter value. Default value is 1.5e+5.")
    parser.add_argument("-doc", "--document_passes", default=1,
                        help="Number of document passes. Default value is 1.")
    parser.add_argument("-col", "--collection_passes", default=25,
                        help="Number of collection passes. Default value is 25.")
    parser.add_argument("-tf", "--topical_files", default=10,
                        help="Number of the most topical files to be saved for each topic.")
    args = parser.parse_args()
    main(args)
