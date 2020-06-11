"""
Parsing-related functionality.
"""
from collections import Counter
import json
from operator import itemgetter
import os
from subprocess import PIPE, Popen
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Set, Tuple

from joblib import cpu_count, delayed, Parallel
from pygments.lexers.haskell import HaskellLexer
from pygments.lexers.jvm import KotlinLexer, ScalaLexer
from pygments.lexers.objective import SwiftLexer
import pygments
from tqdm import tqdm
import tree_sitter

from .parsing.utils import get_parser
from .language_recognition.utils import get_enry
from .subtokenizing import TokenParser

Subtokenizer = TokenParser()

PROCESSES = cpu_count()

SUPPORTED_LANGUAGES = {"tree-sitter": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                       "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"},
                       "pygments": {"Scala", "Swift", "Kotlin", "Haskell"},
                       "classes": {"JavaScript", "Python", "Java", "C++", "Ruby", "TypeScript",
                                   "TSX", "PHP", "C#"},
                       "functions": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                     "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"}}


class RepositoryError(ValueError):
    """
    A special error for catching wrong links to repositories and skipping such repositories.
    """

    def __init__(self, *args):
        ValueError.__init__(self, *args)


def read_file(file: str) -> str:
    """
    Read the contents of the file.
    :param file: the path to the file.
    :return: the contents of the file.
    """
    with open(file) as fin:
        return fin.read()


class TreeSitterParser:
    # Tree-sitter grammars corresponding to a given language.
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

    # Tree-sitter nodes corresponding to identifiers in a given language.
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

    # Tree-sitter nodes corresponding to classes in a given language.
    CLASSES = {"JavaScript": {"class_declaration"},
               "Python": {"class_definition"},
               "Java": {"class_declaration"},
               "C++": {"class_specifier"},
               "Ruby": {"class"},
               "TypeScript": {"class_declaration"},
               "TSX": {"class_declaration"},
               "PHP": {"class_declaration"},
               "C#": {"class_declaration"}}

    # Tree-sitter nodes corresponding to functions in a given language.
    FUNCTIONS = {"JavaScript": {"function", "function_declaration", "method_definition"},
                 "Python": {"function_definition"},
                 "Java": {"constructor_declaration", "method_declaration",
                          "interface_declaration"},
                 "Go": {"function_declaration", "method_declaration"},
                 "C++": {"function_definition"},
                 "Ruby": {"method", "singleton_method"},
                 "TypeScript": {"function", "function_declaration", "method_definition"},
                 "TSX": {"function", "function_declaration", "method_definition"},
                 "PHP": {"function_definition", "method_declaration"},
                 "C#": {"method_declaration", "indexer_declaration", "property_declaration",
                        "constructor_declaration"},
                 "C": {"function_definition"},
                 "Shell": {"function_definition"},
                 "Rust": {"function_item"}}

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
    def traverse_tree(node: tree_sitter.Node, types: Set[str]) -> List[tree_sitter.Node]:
        """
        Run down the AST (DFS) from a given node and gather its children of necessary types.
        :param node: starting Tree-sitter node.
        :param types: the set of types of interest.
        :return: the list of Tree-sitter nodes of necessary types.
        """
        nodes = []
        for child in node.children:
            if child.type in types:
                nodes.append(child)
            if len(child.children) != 0:
                nodes.extend(TreeSitterParser.traverse_tree(child, types))
        return nodes

    @staticmethod
    def get_code_from_node(code: bytes, node: tree_sitter.Node) -> Tuple[str, int, int, int]:
        """
        Given a node of the AST and the code from which this AST was built, return the original
        code corresponding to this node and its parameters: starting byte, starting line,
        starting symbol in line.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :return: tuple (code, starting byte, starting line, starting symbol in line).
        """
        start_byte, end_byte = TreeSitterParser.get_positional_bytes(node)
        code_snippet = code[start_byte:end_byte].decode("utf-8")
        start_line, start_symbol_in_line = node.start_point
        return code_snippet, start_byte, start_line, start_symbol_in_line

    @staticmethod
    def get_tokens_sequence_from_node(code: bytes, node: tree_sitter.Node, lang: str) -> \
            List[Tuple[str, int, int, int]]:
        """
        Given a node of the AST and the code from which this AST was built, gather a list of
        tuples: subtokens of identifiers and their parameters: starting byte, starting line,
        starting symbol in line.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :param lang: the language of code.
        :return: list of tuples (subtoken, starting byte, starting line, starting symbol in line).
        """
        try:
            token_nodes = TreeSitterParser.traverse_tree(node, TreeSitterParser.IDENTIFIERS[lang])
        except RecursionError:
            return []
        tokens_sequence = []
        for token_node in token_nodes:
            token_verbose = TreeSitterParser.get_code_from_node(code, token_node)
            # Currently, each subtoken returns the coordinates of the original token.
            # TODO: fix the subtokenization to account for the change of coordinates.
            subtokens = [(subtoken,) + token_verbose[1:] for subtoken
                         in list(Subtokenizer.process_token(token_verbose[0]))]
            tokens_sequence.extend(subtokens)
        return tokens_sequence

    @staticmethod
    def get_tokens_sequence_from_code(code: str, lang: str) -> List[Tuple[str, int, int, int]]:
        """
        Given the code and its language, gather subtokens of identifiers and their parameters:
        starting byte, starting line, starting symbol in line.
        :param code: source code as a string.
        :param lang: language of the code.
        :return: list of tuples (subtoken, starting byte, starting line, starting symbol in line).
        """
        try:
            assert lang in SUPPORTED_LANGUAGES["tree-sitter"]
        except AssertionError:
            raise ValueError("Unsupported language!")
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        return TreeSitterParser.get_tokens_sequence_from_node(code, root, lang)

    @staticmethod
    def get_tokens_sequence_from_objects(file: str, lang: str, types: Set[str]) -> \
            List[Tuple[str, List[Tuple[str, int, int, int]]]]:
        """
        Given a file, its language and the necessary types of objects (classes or functions),
        gather lists of subtokens of identifiers and their parameters: starting byte, starting
        line, starting symbol in line - for each object. Returns a list of tuples
        ({file_path}#L{starting_line}, list of tuples with subtokens), one tuple per object.
        :param file: the full path to file.
        :param lang: the language of the file.
        :param types: the set of necessary tree-sitter types of the necessary objects.
        :return: a list of tuples: ({file_path}#L{starting_line}, list of tuples with subtokens
                 and their parameters).
        """
        code = read_file(file)
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        try:
            object_nodes = TreeSitterParser.traverse_tree(root, types)
        except RecursionError:
            return [(file, [])]
        object_tokens = []
        for object_node in object_nodes:
            object_tokens.append(("{file}#L{start_line}"
                                  .format(file=file, start_line=object_node.start_point[0] + 1),
                                  TreeSitterParser.get_tokens_sequence_from_node(code, object_node,
                                                                                 lang)))
        return object_tokens

    @staticmethod
    def get_tokens_sequence_from_classes(file: str, lang: str) \
            -> List[Tuple[str, List[Tuple[str, int, int, int]]]]:
        """
        Given a file and its language, gather a lists of subtokens of identifiers and their
        parameters: starting byte, starting line, starting symbol in line within the classes of
        this file. Returns a list of tuples ({file_path}#L{starting_line}, list of tuples with
        subtokens), one tuple per class. In case of errors returns a tuple of the file name
        and an empty list (for bulk processing of files).
        :param file: the full path to file.
        :param lang: the language of the file.
        :return: a list of tuples per class: ({file_path}#L{starting_line},
                 list of tuples with subtokens and their parameters).
        """
        try:
            return TreeSitterParser.get_tokens_sequence_from_objects(file, lang, TreeSitterParser
                                                                     .CLASSES[lang])
        except UnicodeDecodeError:
            return [(file, [])]

    @staticmethod
    def get_tokens_sequence_from_functions(file: str, lang: str) \
            -> List[Tuple[str, List[Tuple[str, int, int, int]]]]:
        """
        Given a file and its language, gather a lists of subtokens of identifiers and their
        parameters: starting byte, starting line, starting symbol in line within the functions of
        this file. Returns a list of tuples ({file_path}#L{starting_line}, list of tuples with
        subtokens), one tuple per function. In case of errors returns a tuple of the file name
        and an empty list (for bulk processing of files).
        :param file: the full path to file.
        :param lang: the language of the file.
        :return: a list of tuples per function: ({file_path}#L{starting_line},
                 list of tuples with subtokens and their parameters).
        """
        try:
            return TreeSitterParser.get_tokens_sequence_from_objects(file, lang, TreeSitterParser
                                                                     .FUNCTIONS[lang])
        except UnicodeDecodeError:
            return [(file, [])]


class PygmentsParser:
    # Pygments lexers corresponding to a given language.
    LEXERS = {"Scala": ScalaLexer(),
              "Swift": SwiftLexer(),
              "Kotlin": KotlinLexer(),
              "Haskell": HaskellLexer()}
    # Pygments token types corresponding to identifiers in a given language.
    IDENTIFIERS = {"Scala": {pygments.token.Name, pygments.token.Keyword.Type},
                   "Swift": {pygments.token.Name},
                   "Kotlin": {pygments.token.Name},
                   "Haskell": {pygments.token.Name, pygments.token.Keyword.Type}}

    @staticmethod
    def get_tokens_sequence_from_code(code: str, lang: str) -> \
            List[Tuple[str, int, int, int]]:
        """
        Given the code and its language, gather subtokens of identifiers and their parameters:
        starting byte, starting line, starting symbol in line.
        :param code: the code to parse.
        :param lang: the language of the code fragment.
        :return: list of tuples (subtoken, starting byte, starting line, starting symbol in line).
        """
        try:
            assert lang in SUPPORTED_LANGUAGES["pygments"]
        except AssertionError:
            raise ValueError("Unsupported language!")
        tokens = []
        for pair in pygments.lex(code, PygmentsParser.LEXERS[lang]):
            if any(pair[0] in sublist for sublist in PygmentsParser.IDENTIFIERS[lang]):
                # TODO: implement indexes for tokens, it's possible in pygments. (0, 0, 0) for now.
                # Currently, each subtoken returns the coordinates of the original token.
                # TODO: fix the subtokenization to account for the change of coordinates.
                subtokens = [(subtoken,) + (0, 0, 0) for subtoken
                             in list(Subtokenizer.process_token(pair[1]))]
                tokens.extend(subtokens)
        return tokens


def get_tokens_sequence_from_code(code: str, lang: str) -> \
        List[Tuple[str, int, int, int]]:
    """
    Given the code and its language, gather subtokens of identifiers and their parameters:
    starting byte, starting line, starting symbol in line.
    :param code: the code to parse.
    :param lang: the language of the code fragment.
    :return: list of tuples (subtoken, starting byte, starting line, starting symbol in line).
    """
    if lang in SUPPORTED_LANGUAGES["tree-sitter"]:
        return TreeSitterParser.get_tokens_sequence_from_code(code, lang)
    elif lang in SUPPORTED_LANGUAGES["pygments"]:
        return PygmentsParser.get_tokens_sequence_from_code(code, lang)
    else:
        raise ValueError("Unsupported language!")


def get_tokens_sequence_from_file(file: str, lang: str) -> \
        List[Tuple[str, List[Tuple[str, int, int, int]]]]:
    """
    Given the file and its language, gather subtokens of identifiers and their parameters:
    starting byte, starting line, starting symbol in line.
    :param file: the full path to file.
    :param lang: the language of the file.
    :return: list of tuples (subtoken, starting byte, starting line, starting symbol in line).
    """
    try:
        code = read_file(file)
        return [(file, get_tokens_sequence_from_code(code, lang))]
    except (FileNotFoundError, UnicodeDecodeError):
        return [(file, [])]


def sequence_to_counter(sequence: List[Tuple[str, int, int, int]]) -> Counter:
    """
    Transforms a list of tuples with tokens and their information into a counter object.
    :param sequence: a list of tuples, where the first element of the tuple is a token.
    :return: a Counter object of the tokens and their counts.
    """
    tokens = []
    for token in sequence:
        tokens.append(token[0])
    return Counter(tokens)


def split_list_into_batches(lst: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a given list into sublists with a given maximum number of items.
    :param lst: a list.
    :param batch_size: the maximum number of items in the sublists.
    :return: a list of lists, splitting the original list into batches.
    """
    return [lst[x:x + batch_size] for x in range(0, len(lst), batch_size)]


def assert_trailing_slash(link: str) -> str:
    """
    Add a trailing slash to a link if there isn't one.
    :param link: link to directory or Web site.
    :return: the same link with a trailing slash.
    """
    link = link.rstrip()
    if link[-1] == "/":
        return link
    else:
        return link + "/"


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


def clone_repository(repository: str, directory: str) -> None:
    """
    Clone a given repository into a folder.
    :param repository: a link to GitHub repository, either HTTP or HTTPs.
    :param directory: path to target directory to clone the repository.
    :return: none.
    """
    if "://" in repository:
        body = repository.split("://")[1]
    else:
        raise RepositoryError(f"{repository} is not a valid link!")
    repository = "https://user:password@" + body
    os.system(f"git clone --quiet --depth 1 {repository} {directory}")


def get_latest_commit(directory: str) -> str:
    """
    Get the current commit hash from the Git directory.
    :param directory: the path to a Git directory.
    :return: commit hash.
    """
    return cmdline(f"cd {directory}; git rev-parse HEAD")


def recognize_languages(directory: str) -> dict:
    """
    Recognize the languages in the directory using Enry and return a dictionary
    {language1: [files], language2: [files], ...}.
    :param directory: the path to the directory.
    :return: dictionary {language1: [files], language2: [files], ...}
    """
    return json.loads(cmdline("{enry_loc} -json -mode files {directory}"
                              .format(enry_loc=get_enry(), directory=directory)))


def transform_files_list(lang2files: Dict[str, str], gran: str,
                         language: str) -> List[Tuple[str, str]]:
    """
    Transform the output of Enry on a directory into a list of tuples (full_path_to_file, lang)
    for supported languages only. Supported languages depend on the granularity and whether one
    specific language was specified.
    :param lang2files: the dictionary output of Enry: {language: [files], ...}.
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param language: the language of parsing. 'all' stands for all the languages available for a
                     given parsing granularity, specific languages refer to themselves.
    :return: a list of tuples (full_path_to_file, lang) for the necessary languages.
    """
    # Get the languages available for a given granularity.
    if gran in ["projects", "files"]:  # Projects and files are supported for all languages.
        langs = SUPPORTED_LANGUAGES["tree-sitter"] | SUPPORTED_LANGUAGES["pygments"]
    elif gran == "classes":
        langs = SUPPORTED_LANGUAGES["classes"]
    elif gran == "functions":
        langs = SUPPORTED_LANGUAGES["functions"]
    else:
        raise ValueError("Incorrect granularity of parsing.")
    # If a specific language was specified, override it
    # and check its availability for a given granularity.
    if language != "all":
        try:
            assert language in SUPPORTED_LANGUAGES["tree-sitter"] | SUPPORTED_LANGUAGES["pygments"]
        except AssertionError:
            raise ValueError("Unsupported language!")
        if language in langs:
            langs = [language]
        else:
            raise ValueError(f"{language} doesn't support {gran} granularity.")
    files = []
    for lang in lang2files.keys():
        if lang in langs:
            for file in lang2files[lang]:
                files.append((file, lang))
    return files


def get_full_path(file: str, directory: str) -> str:
    """
    Get the full path to file from the full path to a directory and a relative path to that
    file in that directory.
    :param file: the relative path to file in a directory.
    :param directory: the full path of a directory.
    :return: the full path to file.
    """
    return os.path.abspath(os.path.join(directory, file))


def tokenize_repository(repository: str, local: bool, gran: str, language: str, pool: Parallel) \
        -> Tuple[str, Dict[str, List[Tuple[str, int, int, int]]]]:
    """
    Tokenize a given repository into bags of tokens with the necessary granularity. Return the
    correct name of the repository for links and a dictionary with bags' names as keys and
    the lists of subtokens and their parameters as values. The bag's name is either a path
    to file or a link to GitHub, with starting lines for functions and classes.
    :param repository: a link to the repository. If "local" is False, a link to GitHub,
                       otherwise - a path to a directory.
    :param local: True if tokenizing in local mode (repository is a path to a direcotry),
                  False if tokenizing in default mode (repository is a GitHub link).
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param language: the language of parsing. 'all' stands for all the languages available for a
                     given parsing granularity, specific languages refer to themselves.
    :param pool: the Parallel class instance for multiprocessing.
    :return: the correct name of the repository for links and a dictionary with bags' names as keys
    and the sequences of tokens and their parameters (starting byte, starting line, starting symbol
    in line) as values. The bag's name is either a path to file or a link to GitHub, with starting
    lines for functions and classes.
    """
    bags2tokens = {}
    repository = assert_trailing_slash(repository)
    with TemporaryDirectory() as td:
        # Determine the correct working directory and its name
        if local:
            directory = repository  # Working directly with a path in the local mode
            try:
                assert os.path.isdir(directory)
            except AssertionError:
                raise RepositoryError(f"{directory} doesn't exist!")
            repository_name = directory
        else:
            directory = td  # Working with a temporary directory in the remote mode
            try:
                clone_repository(repository, directory)
            except RepositoryError:
                raise
            commit = get_latest_commit(directory)
            if gran == "projects":
                # With projects granularity, the link to GitHub is to the entire tree
                repository_name = f"{repository}tree/{commit}/"
            else:
                # With other granularities, the links will be to specific files
                repository_name = f"{repository}blob/{commit}/"
        lang2files = recognize_languages(directory)  # Recognize the languages in the directory
        files = transform_files_list(lang2files, gran, language)
        # Gather the tokens for the correct granularity of parsing
        if gran in ["projects", "files"]:
            chunk_results = pool([delayed(get_tokens_sequence_from_file)
                                  (get_full_path(file[0], directory), file[1])
                                  for file in files])
        elif gran == "classes":
            chunk_results = pool([delayed(TreeSitterParser.get_tokens_sequence_from_classes)
                                  (get_full_path(file[0], directory), file[1])
                                  for file in files])
        elif gran == "functions":
            chunk_results = pool([delayed(TreeSitterParser.get_tokens_sequence_from_functions)
                                  (get_full_path(file[0], directory), file[1])
                                  for file in files])
        for chunk_result in chunk_results:
            for bag in chunk_result:
                if len(bag[1]) != 0:  # Skipping the possible empty bags
                    if local:
                        bags2tokens[bag[0]] = bag[1]
                    else:
                        # Replace the temporary directory with the link to GitHub
                        bags2tokens[repository_name +
                                    os.path.relpath(bag[0], directory)] = bag[1]
    return repository_name, bags2tokens


def save_wabbit(reps2bags: Dict[str, Dict[str, List[Tuple[str, int, int, int]]]],
                mode: str, gran: str, output_dir: str, filename: str) -> None:
    """
    Save the bags of tokens in the Vowpal Wabbit format: one bag per line, in the format
    "name token1:parameters token2:parameters...". When run again, overwrites the data.
    :param reps2bags: a dictionary with repositories names as keys and their bags of tokens as
                      values. The bags are also dictionaries with bags' names as keys and
                      sequences of tokens and their parameters as values.
    :param mode: The mode of parsing. 'counters' returns Counter objects of subtokens and their
                 count, 'sequences' returns full sequences of subtokens and their parameters:
                 starting byte, starting line, starting symbol in line, ending symbol in line.
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param output_dir: full path to the output directory.
    :param filename: the name of the output file.
    :return: none.
    """
    def counter_to_wabbit(tokens: Counter) -> str:
        """
        Transforms a Counter object into a saving format of Wabbit:
        "token1:count1, token2:count2..."
        :param tokens: a Counter object of tokens and their count.
        :return: string "token1:count1, token2:count2..." sorted alphabetically.
        """
        sorted_tokens = sorted(tokens.items(), key=itemgetter(0))
        formatted_tokens = []
        for token in sorted_tokens:
            formatted_tokens.append("{token}:{count}".format(token=token[0], count=str(token[1])))
        return " ".join(formatted_tokens)

    def sequence_to_wabbit(sequence: List[Tuple[str, int, int, int]]) -> str:
        """
        Transforms a sequence of tokens and their parameters into a saving format of Wabbit:
        "token1:parameters token2:parameters...".
        :param sequence: a list of tokens and their parameters - starting byte, starting line,
        starting symbol in line.
        :return: string "token1:parameters token2:parameters..." sorted as in original code.
        """
        formatted_tokens = []
        for token in sequence:
            parameters = ",".join([str(parameter) for parameter in token[1:]])
            formatted_tokens.append("{token}:{parameters}".format(token=token[0],
                                                                  parameters=parameters))
        return " ".join(formatted_tokens)

    with open(os.path.abspath(os.path.join(output_dir, filename)), "w+") as fout:
        # If the granularity is 'projects', all the bags for each project are merged into one.
        # Only Counter mode available (sequence is meaningless for the entire project).
        if gran == "projects":
            for repository_name in reps2bags.keys():
                repository_tokens = Counter()
                for bag_tokens in reps2bags[repository_name].values():
                    repository_tokens += sequence_to_counter(bag_tokens)
                fout.write("{name} {tokens}\n"
                           .format(name=repository_name,
                                   tokens=counter_to_wabbit(repository_tokens)))
        # If the granularity is 'files' or finer, then each bag is saved individually.
        else:
            for repository_name in reps2bags.keys():
                for bag_name in reps2bags[repository_name].keys():
                    if mode == "counters":
                        tokens = counter_to_wabbit(sequence_to_counter(
                                           reps2bags[repository_name][bag_name]))
                    elif mode == "sequences":
                        tokens = sequence_to_wabbit(reps2bags[repository_name][bag_name])
                    fout.write("{name} {tokens}\n"
                               .format(name=bag_name, tokens=tokens))


def tokenize_list_of_repositories(repositories_file: str, output_dir: str, batch_size: int,
                                  mode: str, gran: str, language: str, local: bool,
                                  output_format: str) -> None:
    """
    Given the list of links to repositories, tokenize all the repositories in the list,
    writing them in batches to files in a specified output format.
    :param repositories_file: path to text file with a list of repositories.
    :param output_dir: path to the output directory.
    :param batch_size: the number of repositories to be grouped into a single batch / file.
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param mode: The mode of parsing. 'counters' returns Counter objects of subtokens and their
                 count, 'sequences' returns full sequences of subtokens and their parameters:
                 starting byte, ending byte, starting line, starting symbol in line, ending line,
                 ending symbol in line.
    :param language: the language of parsing. 'all' stands for all the languages available for a
                     given parsing granularity, specific languages refer to themselves.
    :param local: True if tokenizing in local mode (the input file contains paths to directories),
                  False if tokenizing in default mode (the input file contains GitHub links)
    :param output_format: the output format. Possible values: ["wabbit"]
    :return: None.
    """
    try:
        assert gran in {"projects", "files", "classes", "functions"}
    except AssertionError:
        raise ValueError("Incorrect granularity of parsing.")
    try:
        assert output_format in {"wabbit"}
    except AssertionError:
        raise ValueError("Incorrect output format.")
    print(f"Tokenizing the repositories in {mode} mode, with {gran} granularity, "
          f"saving into {output_format} format. Languages: {language}.")
    # Reading the input file and splitting repositories into batches.
    assert os.path.exists(repositories_file)
    with open(repositories_file) as fin:
        repositories_list = fin.read().splitlines()
        repositories_batches = split_list_into_batches(repositories_list, batch_size)
    # Creating the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Processing the repositories
    with Parallel(PROCESSES) as pool:
        # Iterating over batches
        for count_batch, batch in enumerate(repositories_batches):
            print(f"Tokenizing batch {count_batch + 1} out of {len(repositories_batches)}.")
            reps2bags = {}
            filename = f"wabbit_{gran}_{count_batch}.txt"
            # Iterating over repositories in the batch
            for repository in tqdm(batch):
                try:
                    repository_name, bags2tokens = tokenize_repository(repository, local,
                                                                       gran, language, pool)
                except RepositoryError:
                    print(f"{repository} is an incorrect link, skipping...")
                    continue
                reps2bags[repository_name] = bags2tokens
            if len(reps2bags.keys()) != 0:  # Skipping possible empty batches.
                if output_format == "wabbit":
                    save_wabbit(reps2bags, mode, gran, output_dir, filename)
    print("Tokenization successfully completed.")
