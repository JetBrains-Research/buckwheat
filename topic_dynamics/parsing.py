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
from tqdm import tqdm


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
                identifier = code[start:end].decode('utf-8').lower()
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


def uci_format(directory: str, name: str) -> None:
    """
    Transform the temporary file with tokens into the UCI bag-of-words format.
    :param directory: the directory with the dataset.
    :param name: name of the processed dataset.
    :return: None.
    """
    number_of_documents = 0
    number_of_nnz = 0
    set_of_tokens = set()
    # Compile a list of all tokens in the dataset for a sorted list
    with open(os.path.abspath(os.path.join(directory, name + '_tokens.txt')), 'r') as fin:
        for line in fin:
            number_of_documents = number_of_documents + 1
            for token in line.rstrip().split(';')[2].split(','):
                number_of_nnz = number_of_nnz + 1
                set_of_tokens.add(token.split(':')[0])
    number_of_tokens = len(set_of_tokens)
    # Sort the list of tokens, transform them to indexes and write to file
    sorted_list_of_tokens = sorted(list(set_of_tokens))
    sorted_dictionary_of_tokens = {}
    with open(os.path.abspath(os.path.join(directory, 'vocab.' + name + '.txt')), 'w+') as fout:
        for index in range(len(sorted_list_of_tokens)):
            sorted_dictionary_of_tokens[sorted_list_of_tokens[index]] = index + 1
            fout.write(sorted_list_of_tokens[index] + '\n')
    # Compile the second necessary file: NNZ triplets sorted by document
    with open(os.path.abspath(os.path.join(directory, name + '_tokens.txt')), 'r') as fin, open(os.path.abspath(os.path.join(directory, 'docword.' + name + '.txt')), 'w+') as fout:
        fout.write(str(number_of_documents) + '\n' + str(number_of_tokens) + '\n' + str(number_of_nnz) + '\n')
        for line in tqdm(fin):
            file_tokens = line.rstrip().split(';')[2].split(',')
            file_tokens_separated = []
            file_tokens_separated_numbered = []
            for entry in file_tokens:
                file_tokens_separated.append(entry.split(':'))
            for entry in file_tokens_separated:
                file_tokens_separated_numbered.append([sorted_dictionary_of_tokens[entry[0]], int(entry[1])])
            file_tokens_separated_numbered = sorted(file_tokens_separated_numbered, key=itemgetter(0), reverse=False)
            for entry in file_tokens_separated_numbered:
                fout.write(str(line.split(';')[0]) + ' ' + str(entry[0]) + ' ' + str(entry[1]) + '\n')


def slice_and_parse_full_files(repository: str, number: int, delta: int, lang: str, name: str) -> None:
    """
    Split the repository, parse the full files, write the data into a file, transform into the UCI format.
    :param repository: path to the repository to process.
    :param number: the amount of dates.
    :param delta: the time step between dates
    :param lang: language of parsing.
    :param name: name of the dataset (directories with resulting files)
    :return: None.
    """
    # Create a folder for created files
    print('Creating the temporal slices of the data.')
    directory = os.path.abspath(os.path.join(repository, os.pardir, name + '_processed'))
    os.mkdir(directory)
    dates = get_dates(number, delta)
    lists_of_files = {}
    # Create temporal slices of the project and get a list of files for each slice
    for date in tqdm(dates):
        subdirectory = os.path.abspath(os.path.join(directory, date.strftime('%Y-%m-%d')))
        checkout_by_date(repository, subdirectory, date)
        lists_of_files[date.strftime('%Y-%m-%d')] = get_a_list_of_files(subdirectory, get_extensions(lang))
    indexes_of_slices = {}
    count = 0
    # Write the data into a temporary file: by slices, then by documents
    print('Parsing each of the temporal slices.')
    with open(os.path.abspath(os.path.join(directory, name + '_tokens.txt')), 'w+') as fout:
        for date in tqdm(dates):
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
    # Write the index boundaries of slices into a separate log file
    print('Writing the index boundaries of slices into an auxiliary file.')
    with open(os.path.abspath(os.path.join(directory, name + '_slices.txt')), 'w+') as fout:
        for date in indexes_of_slices.keys():
            fout.write(date + ';' + str(indexes_of_slices[date][0]) + ',' + str(indexes_of_slices[date][1]) + '\n')
    print('Transforming the data into the UCI format for topic-modeling.')
    uci_format(directory, name)


def slice_and_parse_diffs(repository: str, number: int, delta: int, lang: str, name: str) -> None:
    """
    Split the repository, parse the full files, extract the diffs, write the data into a file, transform into the UCI format.
    :param repository: path to the repository to process.
    :param number: the amount of dates.
    :param delta: the time step between dates
    :param lang: language of parsing.
    :param name: name of the dataset (directories with resulting files)
    :return: None.
    """
    # Create a folder for created files
    print('Creating the temporal slices of the data.')
    directory = os.path.abspath(os.path.join(repository, os.pardir, name + '_diffs_processed'))
    os.mkdir(directory)
    dates = get_dates(number, delta)
    lists_of_files = {}
    # Create temporal slices of the project and get a list of files for each slice
    for date in tqdm(dates):
        subdirectory = os.path.abspath(os.path.join(directory, date.strftime('%Y-%m-%d')))
        checkout_by_date(repository, subdirectory, date)
        lists_of_files[date.strftime('%Y-%m-%d')] = get_a_list_of_files(subdirectory, get_extensions(lang))
    indexes_of_slices = {}
    count_index = 0
    # Write the data into temporary files
    os.mkdir(os.path.abspath(os.path.join(directory, name + '_tokens')))
    print('Parsing each of the temporal slices.')
    for date in tqdm(range(number)):
        with open(os.path.abspath(os.path.join(directory, name + '_tokens', str(date + 1) + '_tokens.txt')), 'w+') as fout:
            starting_index = count_index + 1
            for file in lists_of_files[dates[date].strftime('%Y-%m-%d')]:
                if os.path.isfile(file):  # TODO: implement a better file-checking mechanism
                    try:
                        identifiers = get_identifiers(file, lang)
                        if len(identifiers) != 0:
                            count_index += 1
                            formatted_identifiers = transform_identifiers(identifiers)
                            fout.write(str(count_index) + ';' + file + ';' + ','.join(formatted_identifiers) + '\n')
                    except UnicodeDecodeError:
                        continue
            ending_index = count_index
            indexes_of_slices[dates[date].strftime('%Y-%m-%d')] = (starting_index, ending_index)
    # Write the index boundaries of slices into a separate log file
    print('Writing the index boundaries of slices into an auxiliary file.')
    with open(os.path.abspath(os.path.join(directory, name + '_slices.txt')), 'w+') as fout:
        for date in indexes_of_slices.keys():
            fout.write(date + ';' + str(indexes_of_slices[date][0]) + ',' + str(indexes_of_slices[date][1]) + '\n')
    # Compare the versions and create a new list of tokens
    print('Calculating the diffs between versions and transforming the token lists.')
    indexes_of_diff_slices = {}
    count_index_diff = 0
    with open(os.path.abspath(os.path.join(directory, name + '_diffs_tokens.txt')), 'w+') as fout:
        for date in tqdm(range(2, number + 1)):
            starting_index_diff = count_index_diff+ 1
            previous_version = {}
            with open(os.path.abspath(os.path.join(directory, name + '_tokens', str(date - 1) + '_tokens.txt')), 'r') as fin:
                for line in fin:
                    previous_version[line.rstrip().split(';')[1]] = line.rstrip().split(';')[2]
            current_version = []
            with open(os.path.abspath(os.path.join(directory, name + '_tokens', str(date) + '_tokens.txt')), 'r') as fin:
                for line in fin:
                    current_version.append(line.rstrip().split(';')[1])
                    address = line.rstrip().split(';')[1]
                    tokens = {}
                    for token in line.rstrip().split(';')[2].split(','):
                        tokens[token.split(':')[0]] = int(token.split(':')[1])
                    tokens = Counter(tokens)
                    old_address = address.replace(dates[date - 1].strftime('%Y-%m-%d'), dates[date - 2].strftime('%Y-%m-%d'), 1)
                    if old_address in previous_version.keys():
                        old_tokens = {}
                        for token in previous_version[old_address].split(','):
                            old_tokens[token.split(':')[0]] = int(token.split(':')[1])
                        old_tokens = Counter(old_tokens)
                        created_tokens = sorted((tokens - old_tokens).items(), key=itemgetter(1), reverse=True)
                        deleted_tokens = sorted((old_tokens - tokens).items(), key=itemgetter(1), reverse=True)
                        new_tokens = []
                        if len(created_tokens) != 0:
                            for token in created_tokens:
                                new_token = '+' + token[0]
                                new_tokens.append([new_token, token[1]])
                        if len(deleted_tokens) != 0:
                            for token in deleted_tokens:
                                new_token = '-' + token[0]
                                new_tokens.append([new_token, token[1]])
                    else:
                        tokens = sorted(tokens.items(), key=itemgetter(1), reverse=True)
                        new_tokens = []
                        for token in tokens:
                            new_token = '+' + token[0]
                            new_tokens.append([new_token, token[1]])
                    if len(new_tokens) != 0:
                        formatted_new_tokens = transform_identifiers(new_tokens)
                        count_index_diff = count_index_diff + 1
                        fout.write(str(count_index_diff) + ';' + address + ';' + ','.join(formatted_new_tokens) + '\n')
            for address in previous_version.keys():
                new_address = address.replace(dates[date - 2].strftime('%Y-%m-%d'), dates[date - 1].strftime('%Y-%m-%d'), 1)
                if new_address not in current_version:
                    new_tokens = []
                    for token in previous_version[address].split(','):
                        new_tokens.append(['-' + token.split(':')[0], int(token.split(':')[1])])
                    formatted_new_tokens = transform_identifiers(new_tokens)
                    count_index_diff = count_index_diff + 1
                    fout.write(str(count_index_diff) + ';' + address + ';' + ','.join(formatted_new_tokens) + '\n')
            ending_index_diff = count_index_diff
            indexes_of_diff_slices[dates[date - 1].strftime('%Y-%m-%d')] = (starting_index_diff, ending_index_diff)

    print('Writing the index boundaries of slices into an auxiliary file (updated).')
    with open(os.path.abspath(os.path.join(directory, name + '_diffs_slices.txt')), 'w+') as fout:
        for date in indexes_of_diff_slices.keys():
            fout.write(date + ';' + str(indexes_of_diff_slices[date][0]) + ',' + str(indexes_of_diff_slices[date][1]) + '\n')

    print('Transforming the data into the UCI format for topic-modeling.')
    uci_format(directory, name + '_diffs')