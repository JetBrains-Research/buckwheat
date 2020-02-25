"""
Parsing-related functionality.
"""

from typing import List, Tuple

import os
import tree_sitter
from .parsers.utils import get_parser
from .slicing import get_dates, checkout_by_date
from collections import Counter
from operator import itemgetter
from glob import glob


def get_extensions(lang: str) -> str:
    """
    Returns the extension for a given language. TODO: more than one extension.
    :param lang: language name.
    :return: the extension.
    """
    extensions = {'cpp': 'cpp',
                  'java': 'java',
                  'python': 'py'}
    return extensions[lang]


def get_a_list_of_files(directory: str, extension: str) -> List[str]:
    """
    Get a list of files with a given extension.
    :param directory: the root directory that is studied.
    :param extension: extension of the listed files.
    :return: list of file paths.
    """
    list_of_files = [y for x in os.walk(directory) for y in glob(os.path.join(x[0], '*.' + extension))]
    return list_of_files


def read_file(file: str) -> bytes:
    """
    Read the contents of the file.
    :param file: address of the file.
    :return: bytes with the contents of the file.
    """
    with open(file, 'r') as fin:
        code = bytes(fin.read(), 'utf-8')
    return code


def get_positional_bytes(node: tree_sitter.Node) -> Tuple[int, int]:
    """
    Extract start and end byte.
    :param node: node on the AST.
    :return: (start byte, end byte)
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
    node_types = {'c': ['identifier', 'type_identifier'],
                  'c-sharp': ['identifier', 'type_identifier'],
                  'cpp': ['identifier', 'type_identifier'],
                  'java': ['identifier', 'type_identifier'],
                  'python': ['identifier', 'type_identifier']}

    def traverse_tree(node: tree_sitter.Node) -> None:
        """
        Run down the AST from a given node and gather identifiers from its childern.
        :param node: starting node.
        :return: None.
        """
        for child in node.children:
            if child.type in node_types[lang]:
                start, end = get_positional_bytes(child)
                identifier = code[start:end].decode('utf-8')
                if '\n' not in identifier:  # Will break output files. Can add other bad characters later
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
    :return: a list of identifiers in the writable for, "identifier:count".
    """
    formatted_identifiers = []
    for identifier in identifiers:
        if identifier[0].rstrip() != '':  # Checking for occurring empty tokens.
            formatted_identifiers.append(identifier[0].rstrip() + ':' + str(identifier[1]).rstrip())
    return formatted_identifiers


def main(repository: str, number: int, delta: int, lang: str, output: str, output_info: str) -> None:
    """
    Split the repository, parse the files, write the data into a file.
    :param repository: path to the repository to process, must have a git file.
    :param number: the amount of dates.
    :param delta: the time step between dates
    :param lang: language of parsing.
    :param output: an output file.
    :param output_info: a file for information about output (slice indexes).
    :return: None.
    """
    directory = os.path.abspath(os.path.join(repository, os.pardir, 'project_slices'))
    os.mkdir(directory)
    dates = get_dates(number, delta)
    lists_of_files = {}
    for date in dates:
        subdirectory = os.path.abspath(os.path.join(directory, date.strftime('%Y-%m-%d')))
        checkout_by_date(repository, subdirectory, date)
        lists_of_files[date.strftime('%Y-%m-%d')] = get_a_list_of_files(subdirectory, get_extensions(lang))
    indexes_of_slices = {}
    count = 0
    with open(output, 'w+') as fout:
        for date in dates:
            starting_index = count + 1
            for file in lists_of_files[date.strftime('%Y-%m-%d')]:
                if os.path.isfile(file):  # TODO: implement a better file-checking mechanism
                    try:
                        identifiers = get_identifiers(file, lang)
                        if len(identifiers) != 0:
                            count += 1
                            formatted_identifiers = transform_identifiers(identifiers)
                            fout.write(str(count) + ';' + file + ';' + ','.join(formatted_identifiers) + '\n')
                    except UnicodeDecodeError:
                        continue
            ending_index = count
            indexes_of_slices[date.strftime('%Y-%m-%d')] = (starting_index, ending_index)
    with open(output_info, 'w+') as fout:
        for date in indexes_of_slices.keys():
            fout.write(date + ';' + str(indexes_of_slices[date][0]) + ',' + str(indexes_of_slices[date][1]) + '\n')
