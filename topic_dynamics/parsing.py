"""
Parsing-related functionality.
"""
from collections import Counter
import json
import numpy as np
from operator import itemgetter
import os
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory
from typing import List, Tuple

from joblib import cpu_count, delayed, Parallel
from tqdm import tqdm
import tree_sitter

from .language_recognition.utils import get_enry
from .parsers.utils import get_parser
from .subtokenizing import TokenParser

processes = cpu_count()

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
    Clone a given repository into a folder.
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
    Extract start and end byte of the tree-sitter Node.
    :param node: node on the AST.
    :return: (start byte, end byte).
    """
    start = node.start_byte
    end = node.end_byte
    return start, end


def get_identifiers(file: str, lang: str) -> Counter:
    """
    Gather a Counter object of identifiers in the file and their count.
    :param file: the path to the file.
    :param lang: the language of file.
    :return: a Counter object of items: identifier and count.
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
    return Counter(identifiers)


def get_identifiers_from_list(files_list: List[str], lang: str) -> Counter:
    """
    Gather the identifiers of all the files in the list into a single Counter object.
    :param files_list: the list of the paths to files.
    :param lang: the language of file.
    :return: a Counter object of items: identifier and count.
    """
    identifiers = Counter()
    for file in files_list:
        try:
            file_identifiers = get_identifiers(file, lang)
            identifiers = identifiers + file_identifiers
        except UnicodeDecodeError:
            continue

    return identifiers


def create_absolute_chunks(directory: str, files_list: List[str]) -> List[List[str]]:
    """
    Given a directory and a list of relative paths to files in this directory,
    create approximately equal lists of full paths to these files.
    :param directory: a full path to the directory.
    :param files_list: a list of relative paths to files in the given directory.
    :return: a list of approximately equal lists with full paths to files.
    """
    n_files = len(files_list)
    if n_files < processes:
        return [[os.path.abspath(os.path.join(directory, file))
                 for file in files_list], [] * (processes - 1)]
    else:
        chunk_size = len(files_list) // processes
        return np.array_split([os.path.abspath(os.path.join(directory, file))
                               for file in files_list], chunk_size)


def transform_identifiers(identifiers: Counter) -> List[str]:
    """
    Transform the original list of identifiers into the writable form.
    :param identifiers: a Counter object of identifiers and their count.
    :return: a list of identifiers in the writable form, "identifier:count".
    """
    sorted_indentifiers = sorted(identifiers.items(), key=itemgetter(1), reverse=True)
    formatted_identifiers = []
    for identifier in sorted_indentifiers:
        if identifier[0].rstrip() != "":  # Checking for occurring empty tokens.
            formatted_identifiers.append("{identifier}:{count}"
                                         .format(identifier=identifier[0].rstrip(),
                                                 count=str(identifier[1]).rstrip()))
    return formatted_identifiers


def tokenize_repositories(repositories_file: str, output_dir: str, batch_size: int) -> None:
    """
    Given the list of links to GitHub, tokenize all the repositories in the list,
    writing them in batches to files, a single repository per line.
    When run several times, overwrites the data.
    :param repositories_file: path to text file with a list of repositories links on GitHub.
    :param output_dir: path to the output directory.
    :param batch_size: the number of repositories to be grouped into a single batch.
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
    for repository in tqdm(repositories_list):
        count_batch = count // batch_size + 1
        with open(os.path.abspath(os.path.join(output_dir, f"docword{count_batch}.txt")), "a+") as fout1, \
                open(os.path.abspath(os.path.join(output_dir, f"vocab{count_batch}.txt")), "a+") as fout2:
            identifiers = Counter()
            with TemporaryDirectory() as td:
                clone_repository(repository, td)
                lang2files = recognize_languages(td)
                for lang in lang2files.keys():
                    if lang in SUPPORTED_LANGUAGES.keys():
                        with Parallel(processes) as pool:
                            identifiers_lang = pool([delayed(get_identifiers_from_list)
                                                     (chunk, SUPPORTED_LANGUAGES[lang])
                                                     for chunk in
                                                     create_absolute_chunks(td, lang2files[lang])])
                        for chunk_result in identifiers_lang:
                            identifiers += chunk_result
            if len(identifiers) != 0:
                count += 1
                formatted_identifiers = transform_identifiers(identifiers)
                fout1.write("{repository_index};{repository};{tokens}\n"
                            .format(repository_index=str(count), repository=repository,
                                    tokens=",".join(formatted_identifiers)))
    print("Tokenization successfully completed.")
    