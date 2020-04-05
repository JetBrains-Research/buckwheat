"""
Parsing-related functionality.
"""
from collections import Counter
import json
from operator import itemgetter
import os
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory
from typing import List, Tuple

from tqdm import tqdm
import tree_sitter

from .language_recognition.utils import get_enry
from .parsers.utils import get_parser
from .subtokenizing import TokenParser

SUPPORTED_LANGUAGES = {"Java": "java",
                       "Python": "python",
                       "C++": "cpp"}

NODE_TYPES = {"c": ["identifier", "type_identifier"],
              "c-sharp": ["identifier", "type_identifier"],
              "cpp": ["identifier", "type_identifier"],
              "java": ["identifier", "type_identifier"],
              "python": ["identifier", "type_identifier"]}


def cmdline(command: str) -> str:
    """
    Execute a given command and catch its stdout.
    :param command: a command to execute.
    :return: stdout.
    """
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0].decode("utf8")


def clone_repository(repository: str, directory: str) -> None:
    """
    Checkout a given repository into a folder.
    :param repository: a link to GitHub repository.
    :param directory: path to target directory to clone the repository.
    :return: None.
    """
    os.system("git clone --quiet --depth 1 {repository} {directory}".format(repository=repository,
                                                                            directory=directory))


def recognize_languages(directory: str) -> dict:
    """
    Recognize the languages in the directory using Enry and return a dictionary
    {langauge1: [files], language2: [files], ...}.
    :param directory: the path to the directory.
    :return: dictionary {langauge1: [files], language2: [files], ...}
    """
    return json.loads(cmdline("{enry_loc} -json -mode files {directory}"
                              .format(enry_loc=get_enry(), directory=directory)))


def read_file(file: str) -> bytes:
    """
    Read the contents of the file.
    :param file: the path to the file.
    :return: bytes with the contents of the file.
    """
    with open(file) as fin:
        return bytes(fin.read(), "utf-8")


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
    :param file: the path to the file.
    :param lang: the language of file.
    :return: a list of tuples, identifier and count.
    """
    content = read_file(file)
    tree = get_parser(lang).parse(content)
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
                identifier = content[start:end].decode("utf-8")
                if "\n" not in identifier:  # Will break output files. TODO: try to recreate bug.
                    subtokens = list(TokenParser().process_token(identifier))
                    identifiers.extend(subtokens)
            if len(child.children) != 0:
                try:
                    traverse_tree(child)
                except RecursionError:
                    continue

    traverse_tree(root)
    sorted_identifiers = sorted(Counter(identifiers).items(), key=itemgetter(1), reverse=True)

    return sorted_identifiers


def transform_identifiers(identifiers: List[Tuple[str, int]]) -> List[str]:
    """
    Transform the original list of identifiers into the writable form.
    :param identifiers: list of tuples, identifier and count.
    :return: a list of identifiers in the writable form, "identifier:count".
    """
    formatted_identifiers = []
    for identifier in identifiers:
        if identifier[0].rstrip() != "":  # Checking for occurring empty tokens.
            formatted_identifiers.append("{identifier}:{count}"
                                         .format(identifier=identifier[0].rstrip(),
                                                 count=str(identifier[1]).rstrip()))
    return formatted_identifiers


def slice_and_parse(repositories_file: str, output_dir: str) -> None:
    """
    Split the repository, parse the full files, write the data into a file.
    Can be called for parsing full files and for parsing diffs only.
    When run several times, overwrites the data.
    :param repositories_file: path to text file with a list of repositories links.
    :param output_dir: path to the output directory.
    :return: None.
    """
    print("Tokenizing the repositories.")
    assert os.path.exists(repositories_file)
    repositories_list = []
    with open(repositories_file) as fin:
        for line in fin:
            repositories_list.append(line.rstrip())
    # Create a folder for created files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    # Create temporal slices of the project, get a list of files for each slice,
    # parse all files, save the tokens
    with open(os.path.abspath(os.path.join(output_dir, "tokens.txt")), "w+") as fout1, \
            open(os.path.abspath(os.path.join(output_dir, "indices.txt")), "w+") as fout2:
        for repository in tqdm(repositories_list):
            start_index = count + 1
            with TemporaryDirectory() as td:
                clone_repository(repository, td)
                lang2files = recognize_languages(td)
                for lang in lang2files.keys():
                    if lang in SUPPORTED_LANGUAGES.keys():
                        for file in lang2files[lang]:
                            try:
                                identifiers = get_identifiers(
                                    os.path.abspath(os.path.join(td, file)),
                                    SUPPORTED_LANGUAGES[lang])
                                if len(identifiers) != 0:
                                    count += 1
                                    formatted_identifiers = transform_identifiers(
                                        identifiers)
                                    fout1.write("{file_index};{file_path};{tokens}\n"
                                                .format(file_index=str(count),
                                                        file_path=repository +
                                                                  os.path.relpath(
                                                                      os.path.abspath(
                                                                          os.path.join(
                                                                              td, file)),
                                                                      td),
                                                        tokens=",".join(
                                                            formatted_identifiers)))
                            except UnicodeDecodeError:
                                continue
            end_index = count
            if end_index >= start_index:  # Skips empty repositories
                fout2.write("{repository};{start_index};{end_index}\n"
                            .format(repository=repository, start_index=str(start_index),
                                    end_index=str(end_index)))
