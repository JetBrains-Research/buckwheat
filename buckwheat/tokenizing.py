"""
Tokenization-related functionality.
"""
import logging
import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Set, Tuple

from joblib import cpu_count, delayed, Parallel
from pygments.lexers.haskell import HaskellLexer
from pygments.lexers.jvm import KotlinLexer, ScalaLexer
from pygments.lexers.objective import SwiftLexer
import pygments
import tree_sitter

from .dataclasses import IdentifierData
from .language_recognition.utils import recognize_languages
from .parsing.utils import get_parser
from .saving import OutputFormats
from .subtokenizing import TokenParser
from .utils import assert_trailing_slash, clone_repository, get_full_path, \
    get_latest_commit, read_file, split_list_into_batches, RepositoryError

Subtokenizer = TokenParser()

PROCESSES = cpu_count()

SUPPORTED_LANGUAGES = {"tree-sitter": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                       "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"},
                       "pygments": {"Scala", "Swift", "Kotlin", "Haskell"},
                       "classes": {"JavaScript", "Python", "Java", "C++", "Ruby", "TypeScript",
                                   "TSX", "PHP", "C#"},
                       "functions": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                     "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"}}


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
    def get_identifier_from_node(code: bytes, node: tree_sitter.Node) -> IdentifierData:
        """
        Given an identifier node of the AST and the code from which this AST was built,
        return the IdentifierData object.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :return: IdentifierData object.
        """
        start_byte, end_byte = TreeSitterParser.get_positional_bytes(node)
        identifier = code[start_byte:end_byte].decode("utf-8")
        start_line, start_column = node.start_point
        return IdentifierData(identifier, start_byte, start_line, start_column)

    @staticmethod
    def get_tokens_sequence_from_node(code: bytes, node: tree_sitter.Node, lang: str,
                                      subtokenize: bool = False) -> List[IdentifierData]:
        """
        Given a node of the AST and the code from which this AST was built, gather a list of
        identifiers in it as IdentifierData objects.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :param lang: the language of code.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: list of IdentifierData objects.
        """
        try:
            token_nodes = TreeSitterParser.traverse_tree(node, TreeSitterParser.IDENTIFIERS[lang])
        except RecursionError:
            return []
        tokens_sequence = []
        for token_node in token_nodes:
            token = TreeSitterParser.get_identifier_from_node(code, token_node)
            # Currently, each subtoken returns the coordinates of the original token.
            # TODO: fix the subtokenization to account for the change of coordinates.
            if not subtokenize:
                token.identifier = token.identifier.lower()
                tokens_sequence.append(token)
            else:
                subtokens = [IdentifierData(identifier=subtoken, start_byte=token.start_byte,
                                            start_line=token.start_line,
                                            start_column=token.start_column)
                             for subtoken in list(Subtokenizer.process_token(token.identifier))]
                tokens_sequence.extend(subtokens)
        return tokens_sequence

    @staticmethod
    def get_tokens_sequence_from_code(code: str, lang: str,
                                      subtokenize: bool = False) -> List[IdentifierData]:
        """
        Given the code and its language, gather identifiers in it as IdentifierData objects.
        :param code: source code as a string.
        :param lang: language of the code.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: list of IdentifierData objects.
        """
        try:
            assert lang in SUPPORTED_LANGUAGES["tree-sitter"]
        except AssertionError:
            raise ValueError("Unsupported language!")
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        return TreeSitterParser.get_tokens_sequence_from_node(code, root, lang, subtokenize)

    @staticmethod
    def get_tokens_sequence_from_objects(file: str, lang: str, types: Set[str],
                                         subtokenize: bool = False) -> \
            List[Tuple[str, List[IdentifierData]]]:
        """
        Given a file, its language and the necessary types of objects (classes or functions),
        gather lists of identifiers as lists of IdentifierData for each object.
        Returns a list of tuples (path to file with lines, list of IdentifierData objects).
        :param file: the full path to file.
        :param lang: the language of the file.
        :param types: the set of necessary tree-sitter types of the necessary objects.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: a list of tuples: ({file_path}#L{start_line}-L{end_line},
                                    list of IdentifierData objects).
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
            object_tokens.append(("{file}#L{start_line}-L{end_line}"
                                  .format(file=file, start_line=object_node.start_point[0] + 1,
                                          end_line=object_node.end_point[0] + 1),
                                  TreeSitterParser
                                  .get_tokens_sequence_from_node(code, object_node,
                                                                 lang, subtokenize)))
        return object_tokens

    @staticmethod
    def get_tokens_sequence_from_classes(file: str, lang: str, subtokenize: bool = False) \
            -> List[Tuple[str, List[IdentifierData]]]:
        """
        Given a file and its language, gather lists of identifiers as lists of IdentifierData for
        each class. Returns a list of tuples (path to file with lines, list of IdentifierData
        objects).
        :param file: the full path to file.
        :param lang: the language of the file.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: a list of tuples, one per class: ({file_path}#L{start_line}-L{end_line},
                                                  list of IdentifierData objects).
        """
        try:
            return TreeSitterParser.get_tokens_sequence_from_objects(file, lang, TreeSitterParser
                                                                     .CLASSES[lang], subtokenize)
        except UnicodeDecodeError:
            return [(file, [])]

    @staticmethod
    def get_tokens_sequence_from_functions(file: str, lang: str, subtokenize: bool = False) \
            -> List[Tuple[str, List[IdentifierData]]]:
        """
        Given a file and its language, gather lists of identifiers as lists of IdentifierData for
        each function. Returns a list of tuples (path to file with lines, list of IdentifierData
        objects).
        :param file: the full path to file.
        :param lang: the language of the file.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: a list of tuples, one per function: ({file_path}#L{start_line}-L{end_line},
                                                      list of IdentifierData objects).
        """
        try:
            return TreeSitterParser.get_tokens_sequence_from_objects(file, lang, TreeSitterParser
                                                                     .FUNCTIONS[lang], subtokenize)
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
    def get_tokens_sequence_from_code(code: str, lang: str,
                                      subtokenize: bool = False) -> List[IdentifierData]:
        """
        Given the code and its language, gather subtokens of identifiers as IdentifierData objects.
        :param code: the code to parse.
        :param lang: the language of the code fragment.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: list of IdentifierData objects.
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
                if not subtokenize:
                    tokens.append(IdentifierData(pair[1].lower(), 0, 0, 0))
                else:
                    subtokens = [IdentifierData(subtoken, 0, 0, 0)
                                 for subtoken in list(Subtokenizer.process_token(pair[1]))]
                    tokens.extend(subtokens)
        return tokens


def get_tokens_sequence_from_code(code: str, lang: str,
                                  subtokenize: bool = False) -> List[IdentifierData]:
    """
    Given the code and its language, gather subtokens of identifiers as IdentifierData objects.
    :param code: the code to parse.
    :param lang: the language of the code fragment.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: list of IdentifierData objects.
    """
    if lang in SUPPORTED_LANGUAGES["tree-sitter"]:
        return TreeSitterParser.get_tokens_sequence_from_code(code, lang, subtokenize)
    elif lang in SUPPORTED_LANGUAGES["pygments"]:
        return PygmentsParser.get_tokens_sequence_from_code(code, lang, subtokenize)
    else:
        raise ValueError("Unsupported language!")


def get_tokens_sequence_from_file(file: str, lang: str, subtokenize: bool = False) -> \
        List[Tuple[str, List[IdentifierData]]]:
    """
    Given the file and its language, gather subtokens of identifiers as IdentifierData objects.
    :param file: the code to parse.
    :param lang: the language of the code fragment.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: list with a single tuple (filename, list of IdentifierData objects).
    """
    try:
        code = read_file(file)
        return [(file, get_tokens_sequence_from_code(code, lang, subtokenize))]
    except (FileNotFoundError, UnicodeDecodeError):
        return [(file, [])]


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


def tokenize_repository(repository: str, local: bool, gran: str, language: str, pool: Parallel,
                        subtokenize: bool = False) -> Tuple[str, Dict[str, List[IdentifierData]]]:
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
    :param subtokenize: if True, will split the tokens into subtokens.
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
            logging.debug(f"Cloning {repository}.")
            directory = td  # Working with a temporary directory in the remote mode
            try:
                clone_repository(repository, directory)
            except RepositoryError:
                raise
            commit = get_latest_commit(directory)
            repository_name = f"{repository}tree/{commit}/"
        logging.debug(f"Recognizing languages is {repository}.")
        lang2files = recognize_languages(directory)  # Recognize the languages in the directory
        files = transform_files_list(lang2files, gran, language)
        # Gather the tokens for the correct granularity of parsing
        logging.debug(f"Parsing files in {repository}.")
        if gran in ["projects", "files"]:
            chunk_results = pool([delayed(get_tokens_sequence_from_file)
                                  (get_full_path(file[0], directory), file[1], subtokenize)
                                  for file in files])
        elif gran == "classes":
            chunk_results = pool([delayed(TreeSitterParser.get_tokens_sequence_from_classes)
                                  (get_full_path(file[0], directory), file[1], subtokenize)
                                  for file in files])
        elif gran == "functions":
            chunk_results = pool([delayed(TreeSitterParser.get_tokens_sequence_from_functions)
                                  (get_full_path(file[0], directory), file[1], subtokenize)
                                  for file in files])
        logging.debug(f"Gathering results for {repository}.")
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


def tokenize_list_of_repositories(repositories_file: str, output_dir: str, batch_size: int,
                                  mode: str, gran: str, language: str, local: bool,
                                  output_format: str, subtokenize: bool = False) -> None:
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
                  False if tokenizing in default mode (the input file contains GitHub links).
    :param output_format: the output format. Possible values: ["wabbit"].
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: None.
    """
    try:
        assert gran in {"projects", "files", "classes", "functions"}
    except AssertionError:
        raise ValueError("Incorrect granularity of parsing.")
    try:
        assert output_format in {"wabbit", "json"}
    except AssertionError:
        raise ValueError("Incorrect output format.")
    logging.info(f"Tokenizing the repositories in {mode} mode, with {gran} granularity, "
                 f"saving into {output_format} format. Languages: {language}. "
                 f"Subtokenizing: {subtokenize}.")
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
            logging.info(f"Tokenizing batch {count_batch + 1} out of {len(repositories_batches)}.")
            reps2bags = {}
            filename = f"{output_format}_{mode}_{gran}_{count_batch}.txt"
            # Iterating over repositories in the batch
            for count_repository, repository in enumerate(batch):
                logging.info(f"Tokenizing repository: {repository} ({count_repository + 1} "
                             f"out of {len(batch)} in batch {count_batch + 1}).")
                try:
                    repository_name, bags2tokens = tokenize_repository(repository, local, gran,
                                                                       language, pool, subtokenize)
                except RepositoryError:
                    logging.warning(f"{repository} is an incorrect link, skipping...")
                    continue
                reps2bags[repository_name] = bags2tokens
            logging.info(f"Writing batch {count_batch + 1} out "
                         f"of {len(repositories_batches)} to file.")
            if len(reps2bags.keys()) != 0:  # Skipping possible empty batches.
                OutputFormats(output_format, reps2bags, mode, gran, output_dir, filename)
            logging.info(f"Finished {count_batch + 1} out of {len(repositories_batches)}.")
    logging.info("Tokenization successfully completed.")
