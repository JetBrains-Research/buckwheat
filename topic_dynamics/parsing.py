"""
Parsing-related functionality.
"""
from collections import Counter
import json
from operator import itemgetter
import os
from pygments.lexers.haskell import HaskellLexer
from pygments.lexers.jvm import KotlinLexer, ScalaLexer
from pygments.lexers.objective import SwiftLexer
import pygments
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

from joblib import cpu_count, delayed, Parallel
import numpy as np
from tqdm import tqdm
import tree_sitter

from .language_recognition.utils import get_enry
from .parsers.utils import get_parser
from .subtokenizing import TokenParser

Subtokenizer = TokenParser()

PROCESSES = cpu_count()

SUPPORTED_LANGUAGES = {"JavaScript": "tree-sitter",
                       "Python": "tree-sitter",
                       "Java": "tree-sitter",
                       "Go": "tree-sitter",
                       "C++": "tree-sitter",
                       "Ruby": "tree-sitter",
                       "TypeScript": "tree-sitter",
                       "TSX": "tree-sitter",
                       "PHP": "tree-sitter",
                       "C#": "tree-sitter",
                       "C": "tree-sitter",
                       "Scala": "pygments",
                       "Shell": "tree-sitter",
                       "Rust": "tree-sitter",
                       "Swift": "pygments",
                       "Kotlin": "pygments",
                       "Haskell": "pygments"}


class TreeSitterParser:
    PARSERS = {"JavaScript": "javascript",
               "Python": "python",
               "Java": "java",
               "Go": "go",
               "C++": "cpp",
               "Ruby": "ruby",
               "TypeScript": "typescript",
               "TSX": "tsx",
               "PHP": "php",
               "C#": "c_sharp",
               "C": "c",
               "Shell": "bash",
               "Rust": "rust"}

    NODE_TYPES = {"JavaScript": {"identifier", "property_identifier",
                                 "shorthand_property_identifier"},
                  "Python": {"identifier"},
                  "Java": {"identifier", "type_identifier"},
                  "Go": {"identifier", "field_identifier", "type_identifier"},
                  "C++": {"identifier", "namespace_identifier", "field_identifier",
                          "type_identifier"},
                  "Ruby": {"identifier", "constant", "symbol"},
                  "TypeScript": {"identifier", "property_identifier",
                                 "shorthand_property_identifier", "type_identifier"},
                  "TSX": {"identifier", "property_identifier",
                          "shorthand_property_identifier", "type_identifier"},
                  "PHP": {"name"},
                  "C#": {"identifier"},
                  "C": {"identifier", "field_identifier", "type_identifier"},
                  "Shell": {"variable_name", "command_name"},
                  "Rust": {"identifier", "field_identifier", "type_identifier"}}

    @staticmethod
    def read_file_bytes(file: str) -> bytes:
        """
        Read the contents of the file.
        :param file: the path to the file.
        :return: bytes with the contents of the file.
        """
        with open(file) as fin:
            return bytes(fin.read(), "utf-8")

    @staticmethod
    def get_positional_bytes(node: tree_sitter.Node) -> Tuple[int, int]:
        """
        Extract start and end byte of the tree-sitter Node.
        :param node: node on the AST.
        :return: (start byte, end byte).
        """
        start = node.start_byte
        end = node.end_byte
        return start, end

    @staticmethod
    def get_tokens(file: str, lang: str) -> Counter:
        """
        Gather a Counter object of tokens in the file and their count.
        :param file: the path to the file.
        :param lang: the language of file.
        :return: a Counter object of items: token and count.
        """
        content = TreeSitterParser.read_file_bytes(file)
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(content)
        root = tree.root_node
        tokens = []

        def traverse_tree(node: tree_sitter.Node) -> None:
            """
            Run down the AST (DFS) from a given node and gather tokens from its children.
            :param node: starting node.
            :return: None.
            """
            for child in node.children:
                if child.type in TreeSitterParser.NODE_TYPES[lang]:
                    start, end = TreeSitterParser.get_positional_bytes(child)
                    token = content[start:end].decode("utf-8")
                    if "\n" not in token:  # Will break output files.
                        subtokens = list(Subtokenizer.process_token(token))
                        tokens.extend(subtokens)
                if len(child.children) != 0:
                    traverse_tree(child)

        try:
            traverse_tree(root)
        except RecursionError:
            return Counter()
        return Counter(tokens)


class PygmentsParser:
    LEXERS = {"Scala": ScalaLexer(),
              "Swift": SwiftLexer(),
              "Kotlin": KotlinLexer(),
              "Haskell": HaskellLexer()}

    TYPES = {"Scala": {pygments.token.Name, pygments.token.Keyword.Type},
             "Swift": {pygments.token.Name},
             "Kotlin": {pygments.token.Name},
             "Haskell": {pygments.token.Name, pygments.token.Keyword.Type}}

    @staticmethod
    def read_file(file: str) -> str:
        """
        Read the contents of the file.
        :param file: the path to the file.
        :return: the contents of the file.
        """
        with open(file) as fin:
            return fin.read()

    @staticmethod
    def get_tokens(file: str, lang: str) -> Counter:
        """
        Gather a Counter object of tokens in the file and their count.
        :param file: the path to the file.
        :param lang: the language of file.
        :return: a Counter object of items: token and count.
        """
        content = PygmentsParser.read_file(file)
        tokens = []
        for pair in pygments.lex(content, PygmentsParser.LEXERS[lang]):
            if any(pair[0] in sublist for sublist in PygmentsParser.TYPES[lang]):
                tokens.extend(list(Subtokenizer.process_token(pair[1])))
        return Counter(tokens)


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
    repository = repository[:8] + "user:password@" + repository[8:]
    os.system("git clone --quiet --depth 1 {repository} {directory}".format(repository=repository,
                                                                            directory=directory))


def recognize_languages(directory: str) -> dict:
    """
    Recognize the languages in the directory using Enry and return a dictionary
        {language1: [files], language2: [files], ...}.
    :param directory: the path to the directory.
    :return: dictionary {language1: [files], language2: [files], ...}
    """
    return json.loads(cmdline("{enry_loc} -json -mode files {directory}"
                              .format(enry_loc=get_enry(), directory=directory)))


def transform_files_list(lang2files: Dict[str, str], directory: str) -> List[Tuple[str, str]]:
    """
    Transform the output of Enry on a directory into a list of tuples (full_path_to_file, lang).
    :param lang2files: the dictionary output of Enry: {language: [files], ...}.
    :param directory: the full path to the directory that was processed with Enry.
    :return: a list of tuples (full_path_to_file, lang) for the supported languages.
    """
    files = []
    for lang in lang2files.keys():
        if lang in SUPPORTED_LANGUAGES.keys():
            for file in lang2files[lang]:
                files.append((os.path.abspath(os.path.join(directory, file)), lang))
    return files


def create_chunks(lst: List[Any]) -> List[List[Any]]:
    """
    Transform a list into approximately equal lists for multiprocessing.
    :param lst: a list.
    :return: a list of approximately equal lists.
    """
    n_files = len(lst)
    if n_files < PROCESSES:
        return [lst, [] * (PROCESSES - 1)]
    else:
        chunk_size = len(lst) // PROCESSES
        return np.array_split(lst, chunk_size)


def get_tokens(file: str, lang: str) -> Counter:
    """
    Gather a Counter object of tokens in the file and their count.
    :param file: the path to the file.
    :param lang: the language of file.
    :return: a Counter object of items: token and count.
    """
    if SUPPORTED_LANGUAGES[lang] == "tree-sitter":
        return TreeSitterParser.get_tokens(file, lang)
    else:
        return PygmentsParser.get_tokens(file, lang)


def get_tokens_from_list(files_list: List[Tuple[str, str]]) -> Tuple[Counter, set]:
    """
    Gather the tokens of all the files in the list into a single Counter object.
    :param files_list: the list of the paths to files and their languages.
    :return: a Counter object of items: token and count.
    """
    tokens = Counter()
    for file in files_list:
        try:
            file_tokens = get_tokens(file[0], file[1])
            tokens = tokens + file_tokens
        except (UnicodeDecodeError, FileNotFoundError):
            continue

    return tokens


def transform_tokens(tokens: Counter, token2number: dict) -> List[str]:
    """
    Transform the original list of tokens into the writable form.
    :param tokens: a Counter object of tokens and their count.
    :param token2number: a dictionary that maps tokens to numbers.
    :return: a list of tokens in the writable form, "n_token:count".
    """
    sorted_tokens = [[token2number[token[0]], token[1]]
                     for token in sorted(tokens.items(), key=itemgetter(1), reverse=True)]
    formatted_tokens = []
    for token in sorted_tokens:
        formatted_tokens.append("{token}:{count}"
                                .format(token=token[0],
                                        count=str(token[1])))
    return formatted_tokens


def tokenize_repositories(repositories_file: str, output_dir: str, batch_size: int) -> None:
    """
    Given the list of links to GitHub, tokenize all the repositories in the list,
    writing them in batches to files, a single repository per line, vocabulary separately.
    When run several times, overwrites the data.
    :param repositories_file: path to text file with a list of repositories links on GitHub.
    :param output_dir: path to the output directory.
    :param batch_size: the number of repositories to be grouped into a single batch.
    :return: None.
    """
    print("Tokenizing the repositories.")
    # Reading the input file and splitting it into batches of necessary size
    assert os.path.exists(repositories_file)
    with open(repositories_file) as fin:
        repositories_list = fin.read().splitlines()
        repositories_batches = [repositories_list[x:x+batch_size]
                                for x in range(0, len(repositories_list), batch_size)]
    # Creating the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Processing the batches
    with Parallel(PROCESSES) as pool:
        # Iterating over batches
        for count_batch, batch in enumerate(repositories_batches):
            print(f"Tokenizing batch {count_batch + 1} out of {len(repositories_batches)}.")
            rep2tokens = {}
            vocab = set()
            # Iterating over repositories in the batch
            for repository in tqdm(batch):
                tokens = Counter()
                with TemporaryDirectory() as td:
                    clone_repository(repository, td)
                    lang2files = recognize_languages(td)
                    files = transform_files_list(lang2files, td)
                    chunk_results = pool([delayed(get_tokens_from_list)(chunk)
                                         for chunk in create_chunks(files)])
                for chunk_result in chunk_results:
                    tokens += chunk_result  # Tokens are unique for every repository
                    vocab.update(chunk_result.keys())  # Vocab is compiled for the entire batch
                if len(tokens) != 0:  # Skipping the possible empty repositories
                    rep2tokens[repository] = tokens
            token2number = {}
            for number, token in enumerate(vocab):
                token2number[token] = number
            # Writing the tokens, one repository per line
            with open(os.path.abspath(os.path.join(output_dir,
                                                   f"docword{count_batch}.txt")), "w+") as fout:
                for repository in rep2tokens.keys():
                    fout.write("{repository};{tokens}\n"
                               .format(repository=repository,
                                       tokens=",".join(transform_tokens(rep2tokens[repository],
                                                                        token2number))))
            # Writing the vocabulary, mapping numbers to tokens
            with open(os.path.abspath(os.path.join(output_dir,
                                                   f"vocab{count_batch}.txt")), "w+") as fout:
                for token in token2number.keys():
                    fout.write("{number};{token}\n".format(number=token2number[token],
                                                           token=token))
    print("Tokenization successfully completed.")
