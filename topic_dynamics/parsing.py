"""
Parsing-related functionality.
"""

from collections import Counter
from tempfile import TemporaryDirectory
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple
import os
import tree_sitter

from tqdm import tqdm

from .parsers.utils import get_parser
from .slicing import get_dates, checkout_by_date

NODE_TYPES = {"c": ["identifier", "type_identifier"],
              "c-sharp": ["identifier", "type_identifier"],
              "cpp": ["identifier", "type_identifier"],
              "java": ["identifier", "type_identifier"],
              "python": ["identifier", "type_identifier"]}


def get_extensions(lang: str) -> str:
    """
    Returns the extension for a given language. TODO: more than one extension.
    :param lang: language name.
    :return: the extension.
    """
    extensions = {"cpp": "cpp",
                  "java": "java",
                  "python": "py"}
    return extensions[lang]


def get_files(directory: str, extension: str) -> List[str]:
    """
    Get a list of files with a given extension.
    :param directory: the root directory that is studied.
    :param extension: extension of the listed files.
    :return: list of file paths.
    """
    dir_path = Path(directory)
    files = [str(path) for path in sorted(dir_path.rglob("*." + extension))]
    return files


def read_file(file: str) -> bytes:
    """
    Read the contents of the file.
    :param file: address of the file.
    :return: bytes with the contents of the file.
    """
    with open(file, "r") as fin:
        code = bytes(fin.read(), "utf-8")
    return code


def get_positional_bytes(node: tree_sitter.Node) -> Tuple[int, int]:
    """
    Extract start and end byte.
    :param node: node on the AST.
    :return: (start byte, end byte).
    """
    start = node.start_byte
    end = node.end_byte
    return start, end


def get_identifiers(file: str, lang: str) -> List[Tuple[str, int]]:
    """
    Gather a sorted list of identifiers in the file and their count.
    :param file: address of the file.
    :param lang: the language of file.
    :return: a list of tuples, identifier and count.
    """
    code = read_file(file)
    tree = get_parser(lang).parse(code)
    root = tree.root_node
    identifiers = []

    def traverse_tree(node: tree_sitter.Node) -> None:
        """
        Run down the AST (DFS) from a given node and gather identifiers from its children.
        :param node: starting node.
        :return: None.
        """
        for child in node.children:
            if child.type in NODE_TYPES[lang]:
                start, end = get_positional_bytes(child)
                identifier = code[start:end].decode("utf-8").lower()
                if "\n" not in identifier:  # Will break output files. Can add other bad characters later
                    identifiers.append(identifier)
            if len(child.children) != 0:
                traverse_tree(child)

    traverse_tree(root)
    sorted_identifiers = sorted(Counter(identifiers).items(), key=itemgetter(1), reverse=True)

    return sorted_identifiers


def transform_identifiers(identifiers: List) -> List[str]:
    """
    Transform the original list of identifiers into the writable form.
    :param identifiers: list of tuples, identifier and count.
    :return: a list of identifiers in the writable form, "identifier:count".
    """
    formatted_identifiers = []
    for identifier in identifiers:
        if identifier[0].rstrip() != "":  # Checking for occurring empty tokens.
            formatted_identifiers.append(identifier[0].rstrip() + ":" + str(identifier[1]).rstrip())
    return formatted_identifiers


def slice_and_parse(repository: str, n_dates: int, time_delta: int, lang: str, name: str) -> None:
    """
    Split the repository, parse the full files, write the data into a file.
    Can be called for parsing full files and for parsing diffs only.
    :param repository: path to the repository to process.
    :param n_dates: the amount of dates.
    :param time_delta: the time step between dates.
    :param lang: language of parsing.
    :param name: name of the dataset (directories with resulting files).
    :return: None.
    """
    print("Creating the temporal slices of the data.")
    # Create a folder for created files
    directory = os.path.abspath(os.path.join(repository, os.pardir, name + "_processed"))
    os.mkdir(directory)
    dates = get_dates(n_dates, time_delta)
    dates_indices = {}
    count = 0
    # Create temporal slices of the project, get a list of files for each slice, parse all files, save the tokens
    with open(os.path.abspath(os.path.join(directory, name + "_tokens.txt")), "w+") as fout:
        for date in tqdm(dates):
            with TemporaryDirectory() as td:
                subdirectory = os.path.abspath(os.path.join(directory, td, date.strftime("%Y-%m-%d")))
                checkout_by_date(repository, subdirectory, date)
                files = get_files(subdirectory, get_extensions(lang))
                starting_index = count + 1
                for file in files:
                    if os.path.isfile(file):  # TODO: implement a better file-checking mechanism
                        try:
                            identifiers = get_identifiers(file, lang)
                            if len(identifiers) != 0:
                                count += 1
                                formatted_identifiers = transform_identifiers(identifiers)
                                fout.write(str(count) + ";" + os.path.relpath(file, os.path.abspath(os.path.join(directory, td))) + ";" + ",".join(formatted_identifiers) + "\n")
                        except UnicodeDecodeError:
                            continue
                ending_index = count
                dates_indices[date.strftime("%Y-%m-%d")] = (starting_index, ending_index)
    # Write the index boundaries of slices into a separate log file
    print("Writing the index boundaries of slices into an auxiliary file.")
    with open(os.path.abspath(os.path.join(directory, name + "_slices.txt")), "w+") as fout:
        for date in dates_indices.keys():
            fout.write(date + ";" + str(dates_indices[date][0]) + "," + str(dates_indices[date][1]) + "\n")


def split_token_file(directory: str, name: str) -> None:
    """
    Split a single file with tokens into splits for calculating diffs.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    print("Splitting the tokens of full files by versions.")
    slice_number = 0
    dates_indices = {}
    os.mkdir(os.path.abspath(os.path.join(directory, name + "_slices_tokens")))
    # Read the data about the indices boundaries of slices
    with open(os.path.abspath(os.path.join(directory, name + "_slices.txt")), "r") as fin:
        for line in fin:
            slice_number = slice_number + 1
            dates_indices[slice_number] = (int(line.rstrip().split(";")[1].split(",")[0]),
                                           int(line.rstrip().split(";")[1].split(",")[1]))
    # Write the tokens of each slice into a separate file, numbered incrementally
    for date in tqdm(dates_indices.keys()):
        with open(os.path.abspath(os.path.join(directory, name + "_tokens.txt")), "r") as fin, open(os.path.abspath(os.path.join(directory, name + "_slices_tokens", str(date) + ".txt")), "w+") as fout:
            for line in fin:
                if (int(line.split(";")[0]) >= dates_indices[date][0]) and (int(line.split(";")[0]) <= dates_indices[date][1]):
                    fout.write(line)


def calculate_diffs(directory: str, name: str, dates: List) -> None:
    """
    For token files in a given directory transform into a single token file with diffs for topic modeling.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :param dates: a list of dates used for slicing.
    :return: None.
    """
    print("Calculating the diffs between versions and transforming the token lists.")
    diff_indices = {}
    count_index_diff = 0
    with open(os.path.abspath(os.path.join(directory, name + "_diffs_tokens.txt")), "w+") as fout:
        for date in tqdm(range(2, len(dates) + 1)):
            starting_index_diff = count_index_diff + 1
            # Save the tokens of the "previous" slice into memory
            previous_version = {}
            with open(os.path.abspath(os.path.join(directory, name + "_slices_tokens", str(date - 1) + ".txt")),
                      "r") as fin:
                for line in fin:
                    previous_version[line.rstrip().split(";")[1]] = line.rstrip().split(";")[2]
            current_version = []
            with open(os.path.abspath(os.path.join(directory, name + "_slices_tokens", str(date) + ".txt")),
                      "r") as fin:
                for line in fin:
                    # Iterate over files in the "current" version
                    current_version.append(line.rstrip().split(";")[1])
                    address = line.rstrip().split(";")[1]
                    tokens = {}
                    for token in line.rstrip().split(";")[2].split(","):
                        tokens[token.split(":")[0]] = int(token.split(":")[1])
                    tokens = Counter(tokens)
                    old_address = address.replace(dates[date - 1].strftime("%Y-%m-%d"),
                                                  dates[date - 2].strftime("%Y-%m-%d"), 1)
                    # Check if the file with this name existed in the previous version
                    if old_address in previous_version.keys():
                        old_tokens = {}
                        for token in previous_version[old_address].split(","):
                            old_tokens[token.split(":")[0]] = int(token.split(":")[1])
                        old_tokens = Counter(old_tokens)
                        # Calcualate which tokens have been added and removed between versions
                        created_tokens = sorted((tokens - old_tokens).items(), key=itemgetter(1), reverse=True)
                        deleted_tokens = sorted((old_tokens - tokens).items(), key=itemgetter(1), reverse=True)
                        new_tokens = []
                        if len(created_tokens) != 0:
                            for token in created_tokens:
                                new_token = "+" + token[0]
                                new_tokens.append([new_token, token[1]])
                        if len(deleted_tokens) != 0:
                            for token in deleted_tokens:
                                new_token = "-" + token[0]
                                new_tokens.append([new_token, token[1]])
                    # If the file is new, all of its tokens are considered created
                    else:
                        tokens = sorted(tokens.items(), key=itemgetter(1), reverse=True)
                        new_tokens = []
                        for token in tokens:
                            new_token = "+" + token[0]
                            new_tokens.append([new_token, token[1]])
                    if len(new_tokens) != 0:
                        formatted_new_tokens = transform_identifiers(new_tokens)
                        count_index_diff = count_index_diff + 1
                        fout.write(str(count_index_diff) + ";" + address + ";" + ",".join(formatted_new_tokens) + "\n")
            # Iterate over files in the "previous" version to see which have been deleted
            for address in previous_version.keys():
                new_address = address.replace(dates[date - 2].strftime("%Y-%m-%d"),
                                              dates[date - 1].strftime("%Y-%m-%d"), 1)
                if new_address not in current_version:
                    new_tokens = []
                    for token in previous_version[address].split(","):
                        new_tokens.append(["-" + token.split(":")[0], int(token.split(":")[1])])
                    formatted_new_tokens = transform_identifiers(new_tokens)
                    count_index_diff = count_index_diff + 1
                    fout.write(str(count_index_diff) + ";" + address + ";" + ",".join(formatted_new_tokens) + "\n")
            ending_index_diff = count_index_diff
            diff_indices[dates[date - 1].strftime("%Y-%m-%d")] = (starting_index_diff, ending_index_diff)
    # Write the index boundaries of slices into a separate log file
    print("Writing the index boundaries of slices into an auxiliary file (updated).")
    with open(os.path.abspath(os.path.join(directory, name + "_diffs_slices.txt")), "w+") as fout:
        for date in diff_indices.keys():
            fout.write(date + ";" + str(diff_indices[date][0]) + "," + str(diff_indices[date][1]) + "\n")


def uci_format(directory: str, name: str) -> None:
    """
    Transform the file with tokens into the UCI bag-of-words format.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    print("Transforming the data into the UCI format for topic-modeling.")
    number_of_documents = 0
    number_of_nnz = 0
    set_of_tokens = set()
    # Compile a list of all tokens in the dataset for a sorted list
    with open(os.path.abspath(os.path.join(directory, name + "_tokens.txt")), "r") as fin:
        for line in fin:
            number_of_documents = number_of_documents + 1
            for token in line.rstrip().split(";")[2].split(","):
                number_of_nnz = number_of_nnz + 1
                set_of_tokens.add(token.split(":")[0])
    number_of_tokens = len(set_of_tokens)
    # Sort the list of tokens, transform them to indexes and write to file
    sorted_list_of_tokens = sorted(list(set_of_tokens))
    sorted_dictionary_of_tokens = {}
    with open(os.path.abspath(os.path.join(directory, "vocab." + name + ".txt")), "w+") as fout:
        for index in range(len(sorted_list_of_tokens)):
            sorted_dictionary_of_tokens[sorted_list_of_tokens[index]] = index + 1
            fout.write(sorted_list_of_tokens[index] + "\n")
    # Compile the second necessary file: NNZ triplets sorted by document
    with open(os.path.abspath(os.path.join(directory, name + "_tokens.txt")), "r") as fin, open(os.path.abspath(os.path.join(directory, "docword." + name + ".txt")), "w+") as fout:
        fout.write(str(number_of_documents) + "\n" + str(number_of_tokens) + "\n" + str(number_of_nnz) + "\n")
        for line in tqdm(fin):
            file_tokens = line.rstrip().split(";")[2].split(",")
            file_tokens_separated = []
            file_tokens_separated_numbered = []
            for entry in file_tokens:
                file_tokens_separated.append(entry.split(":"))
            for entry in file_tokens_separated:
                file_tokens_separated_numbered.append([sorted_dictionary_of_tokens[entry[0]], int(entry[1])])
            file_tokens_separated_numbered = sorted(file_tokens_separated_numbered, key=itemgetter(0), reverse=False)
            for entry in file_tokens_separated_numbered:
                fout.write(str(line.split(";")[0]) + " " + str(entry[0]) + " " + str(entry[1]) + "\n")


def slice_and_parse_full_files(repository: str, n_dates: int, time_delta: int, lang: str, name: str) -> None:
    """
    Split the repository, parse the full files, write the data into a file, transform into the UCI format.
    :param repository: path to the repository to process.
    :param n_dates: the amount of dates.
    :param time_delta: the time step between dates.
    :param lang: language of parsing.
    :param name: name of the dataset (directories with resulting files).
    :return: None.
    """
    directory = os.path.abspath(os.path.join(repository, os.pardir, name + "_processed"))
    slice_and_parse(repository, n_dates, time_delta, lang, name)
    uci_format(directory, name)


def slice_and_parse_diffs(repository: str, n_dates: int, time_delta: int, lang: str, name: str) -> None:
    """
    Split the repository, parse the full files, extract the diffs,
    write the data into a file, transform into the UCI format.
    :param repository: path to the repository to process.
    :param n_dates: the amount of dates.
    :param time_delta: the time step between dates.
    :param lang: language of parsing.
    :param name: name of the dataset (directories with resulting files).
    :return: None.
    """
    directory = os.path.abspath(os.path.join(repository, os.pardir, name + "_processed"))
    dates = get_dates(n_dates, time_delta)
    slice_and_parse(repository, n_dates, time_delta, lang, name)
    split_token_file(directory, name)
    calculate_diffs(directory, name, dates)
    uci_format(directory, name + "_diffs")
