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

from .language_recognition.utils import get_enry
from .parsers.utils import get_parser
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
    # Tree-sitter parsers corresponding to a given language.
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
    def get_code_from_node(code: bytes, node: tree_sitter.Node) -> str:
        """
        Given a node of the AST and the code from which this AST was built, return the original
        code corresponding to this node.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :return: code corresponding to the node.
        """
        start, end = TreeSitterParser.get_positional_bytes(node)
        return code[start:end].decode("utf-8")

    @staticmethod
    def get_tokens_from_node(code: bytes, node: tree_sitter.Node, lang: str) -> Counter:
        """
        Given a node of the AST and the code from which this AST was built, gather
        a Counter object of all the identifiers from this node and all of its children.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :param lang: the language of code.
        :return: a Counter object of items: token and count.
        """
        try:
            token_nodes = TreeSitterParser.traverse_tree(node, TreeSitterParser.IDENTIFIERS[lang])
        except RecursionError:
            return Counter()
        tokens = []
        for token_node in token_nodes:
            token = TreeSitterParser.get_code_from_node(code, token_node)
            subtokens = list(Subtokenizer.process_token(token))
            tokens.extend(subtokens)
        return Counter(tokens)

    @staticmethod
    def get_tokens_from_code(code: str, lang: str) -> Counter:
        """
        Gather a Counter object of tokens in the code fragment and their count.
        :param code: the code to parse.
        :param lang: the language of the code fragment.
        :return: a Counter object of items: token and count.
        """
        try:
            assert lang in SUPPORTED_LANGUAGES["tree-sitter"]
        except AssertionError:
            raise ValueError("Unsupported language!")
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        return TreeSitterParser.get_tokens_from_node(code, root, lang)

    @staticmethod
    def get_tokens_from_objects(file: str, lang: str,
                                types: Set[str]) -> List[Tuple[str, Counter]]:
        """
        Given a file, its language and the necessary types of objects (classes or functions),
        gather a Counter objects with identifiers and their count within these objects in these
        files. Returns a list of tuples ({file_path}#L{starting_line}, Counter),
        one tuple per object.
        :param file: the full path to file.
        :param lang: the language of the file.
        :param types: the set of necessary tree-sitter types of the necessary objects.
        :return: a list of tuples: ({file_path}#L{starting_line}, Counter).
        """
        code = read_file(file)
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        try:
            object_nodes = TreeSitterParser.traverse_tree(root, types)
        except RecursionError:
            return [(file, Counter())]
        object_tokens = []
        for object_node in object_nodes:
            object_tokens.append(("{file}#L{start_line}"
                                  .format(file=file, start_line=object_node.start_point[0] + 1),
                                 TreeSitterParser.get_tokens_from_node(code, object_node, lang)))
        return object_tokens

    @staticmethod
    def get_tokens_from_classes(file: str, lang: str) -> List[Tuple[str, Counter]]:
        """
        Given a file and its language, gather a Counter objects with identifiers and their count
        within the classes of this file. Returns a list of tuples ({file_path}#L{starting_line},
        Counter), one tuple per class.
        :param file: the full path to file.
        :param lang: the language of the file.
        :return: a list of tuples per every class: ({file_path}#L{starting_line}, Counter).
        """
        try:
            return TreeSitterParser.get_tokens_from_objects(file, lang,
                                                            TreeSitterParser.CLASSES[lang])
        except UnicodeDecodeError:
            return [(file, Counter())]

    @staticmethod
    def get_tokens_from_functions(file: str, lang: str) -> List[Tuple[str, Counter]]:
        """
        Given a file and its language, gather a Counter objects with identifiers and their count
        within the functions of this file. Returns a list of tuples ({file_path}#L{starting_line},
        Counter), one tuple per function.
        :param file: the full path to file.
        :param lang: the language of the file.
        :return: a list of tuples per every function: ({file_path}#L{starting_line}, Counter).
        """
        try:
            return TreeSitterParser.get_tokens_from_objects(file, lang,
                                                            TreeSitterParser.FUNCTIONS[lang])
        except UnicodeDecodeError:
            return [(file, Counter())]


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
    def get_tokens_from_code(code: str, lang: str) -> Counter:
        """
        Gather a Counter object of tokens in the code fragment and their count.
        :param code: the code to parse.
        :param lang: the language of the code fragment.
        :return: a Counter object of items: token and count.
        """
        try:
            assert lang in SUPPORTED_LANGUAGES["pygments"]
        except AssertionError:
            raise ValueError("Unsupported language!")
        tokens = []
        for pair in pygments.lex(code, PygmentsParser.LEXERS[lang]):
            if any(pair[0] in sublist for sublist in PygmentsParser.IDENTIFIERS[lang]):
                tokens.extend(list(Subtokenizer.process_token(pair[1])))
        return Counter(tokens)


def get_tokens_from_code(code: str, lang: str) -> Counter:
    """
    Gather a Counter object of tokens in the code fragment and their count.
    :param code: the code to parse.
    :param lang: the language of the code fragment.
    :return: a Counter object of items: token and count.
    """
    if lang in SUPPORTED_LANGUAGES["tree-sitter"]:
        return TreeSitterParser.get_tokens_from_code(code, lang)
    elif lang in SUPPORTED_LANGUAGES["pygments"]:
        return PygmentsParser.get_tokens_from_code(code, lang)
    else:
        raise ValueError("Unsupported language!")


def get_tokens_from_file(file: str, lang: str) -> List[Tuple[str, Counter]]:
    """
    Gather a Counter object of tokens in the file and their count,
    return a tuple (file, Counter(token, count)).
    :param file: the full path to the file.
    :param lang: the language of code.
    :return: the list with a single tuple (file, Counter(token, count)).
    """
    try:
        code = read_file(file)
        return [(file, get_tokens_from_code(code, lang))]
    except (FileNotFoundError, UnicodeDecodeError):
        return [(file, Counter())]


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


def tokenize_repository(repository: str, local: bool, gran: str, language: str,
                        pool: Parallel) -> Tuple[str, Dict[str, Counter]]:
    """
    Tokenize a given repository into bags of tokens with the necessary granularity. Return the
    correct name of the repository for links and a dictionary with bags' names as keys and
    the Counter objects of tokens and their counts as values. The bag's name is either a path
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
    and the Counter objects of tokens and their counts as values. The bag's name is either a path
    to file or a link to GitHub, with starting lines for functions and classes.
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
            chunk_results = pool([delayed(get_tokens_from_file)
                                  (get_full_path(file[0], directory), file[1])
                                  for file in files])
        elif gran == "classes":
            chunk_results = pool([delayed(TreeSitterParser.get_tokens_from_classes)
                                  (get_full_path(file[0], directory), file[1])
                                  for file in files])
        elif gran == "functions":
            chunk_results = pool([delayed(TreeSitterParser.get_tokens_from_functions)
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


def transform_tokens(tokens: Counter) -> List[str]:
    """
    Transform the Counter object of tokens and their count into the writable form.
    :param tokens: a Counter object of tokens and their count.
    :return: a list of tokens in the writable form, "n_token:count", sorted alphabetically.
    """
    sorted_tokens = sorted(tokens.items(), key=itemgetter(0))
    formatted_tokens = []
    for token in sorted_tokens:
        formatted_tokens.append("{token}:{count}"
                                .format(token=token[0],
                                        count=str(token[1])))
    return formatted_tokens


def save_wabbit(reps2bags: Dict[str, Dict[str, Counter]], gran: str,
                output_dir: str, filename: str) -> None:
    """
    Save the bags of tokens in the Vowpal Wabbit format: one bag per line, in the format
    "name token1:count1 token2:count2...". When run again, overwrites the data.
    :param reps2bags: a dictionary with repositories names as keys and their bags of tokens as
                      values. The bags are also dictionaries with bags' names as keys and
                      Counters of tokens and their count as values.
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param output_dir: full path to the output directory.
    :param filename: the name of the output file.
    :return: none.
    """
    with open(os.path.abspath(os.path.join(output_dir, filename)), "w+") as fout:
        # If the granularity is 'projects', all the bags for each project are merged into one.
        if gran == "projects":
            for repository_name in reps2bags.keys():
                repository_tokens = Counter()
                for bag_tokens in reps2bags[repository_name].values():
                    repository_tokens += bag_tokens
                fout.write("{name} {tokens}\n"
                           .format(name=repository_name,
                                   tokens=" ".join(transform_tokens(repository_tokens))))
        # If the granularity is 'files' or finer, then each bag is saved individually.
        else:
            for repository_name in reps2bags.keys():
                for bag_name in reps2bags[repository_name].keys():
                    fout.write("{name} {tokens}\n"
                               .format(name=bag_name,
                                       tokens=" ".join(transform_tokens(
                                           reps2bags[repository_name][bag_name]))))


def tokenize_list_of_repositories(repositories_file: str, output_dir: str, batch_size: int,
                                  gran: str, language: str, local: bool,
                                  output_format: str) -> None:
    """
    Given the list of links to repositories, tokenize all the repositories in the list,
    writing them in batches to files in a specified output format.
    :param repositories_file: path to text file with a list of repositories.
    :param output_dir: path to the output directory.
    :param batch_size: the number of repositories to be grouped into a single batch / file.
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
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
    print(f"Tokenizing the repositories with {gran} granularity, "
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
                    save_wabbit(reps2bags, gran, output_dir, filename)
    print("Tokenization successfully completed.")
