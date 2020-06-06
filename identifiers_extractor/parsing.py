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
from typing import Dict, List, Tuple

from joblib import cpu_count, delayed, Parallel
from tqdm import tqdm
import tree_sitter

from .language_recognition.utils import get_enry
from .parsers.utils import get_parser
from .subtokenizing import TokenParser

Subtokenizer = TokenParser()

PROCESSES = cpu_count()

SUPPORTED_LANGUAGES = {"tree-sitter": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                       "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"},
                       "pygments": {"Scala", "Swift", "Kotlin", "Haskell"},
                       "classes": {},
                       "functions": {}}


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

    IDENTIFIERS = {"JavaScript": {"identifier", "property_identifier",
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
    def get_tokens(code: str, lang: str) -> Counter:
        """
        Gather a Counter object of tokens in the file and their count.
        :param code: the code to parse.
        :param lang: the language of file.
        :return: a Counter object of items: token and count.
        """
        try:
            code = bytes(code, "utf-8")
        except UnicodeDecodeError:
            return Counter()
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        tokens = []

        def traverse_tree(node: tree_sitter.Node) -> None:
            """
            Run down the AST (DFS) from a given node and gather tokens from its children.
            :param node: starting node.
            :return: None.
            """
            for child in node.children:
                if child.type in TreeSitterParser.IDENTIFIERS[lang]:
                    start, end = TreeSitterParser.get_positional_bytes(child)
                    token = code[start:end].decode("utf-8")
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

    IDENTIFIERS = {"Scala": {pygments.token.Name, pygments.token.Keyword.Type},
                   "Swift": {pygments.token.Name},
                   "Kotlin": {pygments.token.Name},
                   "Haskell": {pygments.token.Name, pygments.token.Keyword.Type}}

    @staticmethod
    def get_tokens(code: str, lang: str) -> Counter:
        """
        Gather a Counter object of tokens in the file and their count.
        :param code: the code to parse.
        :param lang: the language of file.
        :return: a Counter object of items: token and count.
        """
        tokens = []
        for pair in pygments.lex(code, PygmentsParser.LEXERS[lang]):
            if any(pair[0] in sublist for sublist in PygmentsParser.IDENTIFIERS[lang]):
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
    return process.communicate()[0].decode("utf8").rstrip()


def clone_repository(repository: str, directory: str) -> str:
    """
    Clone a given repository into a folder.
    :param repository: a link to GitHub repository, either HTTP or HTTPs.
    :param directory: path to target directory to clone the repository.
    :return: the name of the default branch.
    """
    if "://" in repository:
        body = repository.split("://")[1]
    else:
        raise ValueError("{repository} is not a valid link!".format(repository=repository))
    repository = "https://user:password@" + body
    os.system("git clone --quiet --depth 1 {repository} {directory}".format(repository=repository,
                                                                            directory=directory))
    return cmdline("cd {directory}; git rev-parse --abbrev-ref HEAD".format(directory=directory))


def recognize_languages(directory: str) -> dict:
    """
    Recognize the languages in the directory using Enry and return a dictionary
        {language1: [files], language2: [files], ...}.
    :param directory: the path to the directory.
    :return: dictionary {language1: [files], language2: [files], ...}
    """
    return json.loads(cmdline("{enry_loc} -json -mode files {directory}"
                              .format(enry_loc=get_enry(), directory=directory)))


def transform_files_list(lang2files: Dict[str, str], gran: str) -> List[Tuple[str, str]]:
    """
    Transform the output of Enry on a directory into a list of tuples (full_path_to_file, lang).
    :param lang2files: the dictionary output of Enry: {language: [files], ...}.
    :param gran: the granularity of parsing.
    :return: a list of tuples (full_path_to_file, lang) for the necessary languages.
    """
    if gran == "projects" or gran == "files":
        langs = SUPPORTED_LANGUAGES["tree-sitter"] | SUPPORTED_LANGUAGES["pygments"]
    elif gran == "classes":
        langs = SUPPORTED_LANGUAGES["classes"]
    elif gran == "functions":
        langs = SUPPORTED_LANGUAGES["functions"]
    else:
        raise ValueError("Incorrect granularity of parsing.")
    files = []
    for lang in lang2files.keys():
        if lang in langs:
            for file in lang2files[lang]:
                files.append((file, lang))
    return files


def read_file(file: str) -> str:
    """
    Read the contents of the file.
    :param file: the path to the file.
    :return: the contents of the file.
    """
    with open(file) as fin:
        return fin.read()


def get_full_path(file: str, directory: str) -> str:
    """
    Get the full path to file from the full path to a directory and a relative path to that
    file in that directory.
    :param file: the relative path to file in a directory.
    :param directory: the full path of a directory.
    :return: the full path to file.
    """
    return os.path.abspath(os.path.join(directory, file))


def get_tokens(code: str, lang: str) -> Counter:
    """
    Gather a Counter object of tokens in the file and their count.
    :param code: the code to parse.
    :param lang: the language of code.
    :return: a Counter object of items: token and count.
    """
    if lang in SUPPORTED_LANGUAGES["tree-sitter"]:
        return TreeSitterParser.get_tokens(code, lang)
    elif lang in SUPPORTED_LANGUAGES["pygments"]:
        return PygmentsParser.get_tokens(code, lang)
    else:
        raise ValueError("Unknown language.")


def get_tokens_from_file(file: str, lang: str) -> Tuple[str, Counter]:
    """
    Gather a Counter object of tokens in the file and their count,
    return a tuple (file, Counter(token, count)).
    :param file: the full path to the file.
    :param lang: the language of code.
    :return: tuple (file, Counter(token, count)).
    """
    try:
        code = read_file(file)
    except FileNotFoundError:
        return file, Counter()
    return file, get_tokens(code, lang)


def transform_tokens(tokens: Counter) -> List[str]:
    """
    Transform the original list of tokens into the writable form.
    :param tokens: a Counter object of tokens and their count.
    :return: a list of tokens in the writable form, "n_token:count".
    """
    sorted_tokens = sorted(tokens.items(), key=itemgetter(0))
    formatted_tokens = []
    for token in sorted_tokens:
        formatted_tokens.append("{token}:{count}"
                                .format(token=token[0],
                                        count=str(token[1])))
    return formatted_tokens


def save_wabbit(bags: List[Tuple[str, Counter]], output_dir: str, filename: str) -> None:
    """
    Save the bags of tokens in the Vowpal Wabbit format: one bag per line, in the format
    "name token1:count1 token2:count2...".
    :param bags: bags of tokens that are being saved, as a list of
                 tuples (name, Counter(token, count)).
    :param output_dir: full path to the output directory.
    :param filename: the name of the output file.
    :return: none.
    """
    with open(os.path.abspath(os.path.join(output_dir, filename)), "a+") as fout:
        for bag in bags:
            fout.write("{bag} {tokens}\n"
                       .format(bag=bag[0],
                               tokens=" ".join(transform_tokens(bag[1]))))


def tokenize_repositories(repositories_file: str, output_dir: str, gran: str, local: bool) -> None:
    """
    Given the list of links to repositories, tokenize all the repositories in the list,
    writing them in batches to files, a single repository per line, vocabulary separately.
    When run several times, overwrites the data.
    :param repositories_file: path to text file with a list of repositories.
    :param output_dir: path to the output directory.
    :param gran: the granularity of parsing.
    :param local: True if tokenizing in local mode (the input file contains paths to directories),
                  False if tokenizing in default mode (the input file contains GitHub links)
    :return: None.
    """
    assert gran in {"projects", "files", "classes", "functions"}
    print(f"Tokenizing the repositories with {gran} granularity.")
    # Reading the input file
    assert os.path.exists(repositories_file)
    with open(repositories_file) as fin:
        repositories_list = fin.read().splitlines()
    # Creating the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Processing the repositories
    with Parallel(PROCESSES) as pool:
        # Iterating over repositories
        for repository in tqdm(repositories_list):
            files2tokens = {}
            with TemporaryDirectory() as td:
                if local:
                    directory = repository
                    try:
                        assert os.path.isdir(directory)
                    except AssertionError:
                        print("{directory} doesn't exist!".format(directory=directory))
                        continue
                    repository_name = directory
                else:
                    directory = td
                    try:
                        branch = clone_repository(repository, directory)
                    except ValueError:
                        print("{repository} is not a valid link!"
                              .format(repository=repository))
                        continue
                    if gran == "projects":
                        repository_name = repository
                    else:
                        repository_name = "{repository}/blob/{branch}/".format(
                            repository=repository, branch=branch)
                lang2files = recognize_languages(directory)
                files = transform_files_list(lang2files, gran)
                chunk_results = pool([delayed(get_tokens_from_file)
                                      (get_full_path(file[0], directory), file[1])
                                      for file in files])
                for chunk_result in chunk_results:
                    if len(chunk_result[1]) != 0:  # Skipping the possible empty files
                        if local:
                            files2tokens[chunk_result[0]] = chunk_result[1]
                        else:
                            files2tokens[repository_name +
                                         os.path.relpath(chunk_result[0],
                                                         directory)] = chunk_result[1]
            if gran == "files":
                save_wabbit(list(files2tokens.items()), output_dir, f"wabbit_{gran}.txt")
            elif gran == "projects":
                repository_tokens = Counter()
                for file_tokens in files2tokens.values():
                    repository_tokens += file_tokens
                save_wabbit([(repository_name, repository_tokens)],
                            output_dir, f"wabbit_{gran}.txt")
    print("Tokenization successfully completed.")
