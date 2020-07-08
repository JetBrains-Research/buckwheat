"""
Tokenization-related functionality.
"""
import logging
import os
from collections import deque
from tempfile import TemporaryDirectory
from typing import Iterator, List, Optional, Set, Tuple, Union

import pygments
import tree_sitter
from joblib import cpu_count, delayed, Parallel
from pygments.lexers.haskell import HaskellLexer
from pygments.lexers.jvm import KotlinLexer, ScalaLexer
from pygments.lexers.objective import SwiftLexer

from .language_recognition.utils import recognize_languages_dir
from .parsing.utils import get_parser
from .saver import OutputFormats
from .subtokenizer import TokenParser
from .utils import SUPPORTED_LANGUAGES, PARSING_MODES, GRANULARITIES, OUTPUT_FORMATS, \
    IdentifiersTypes, ObjectTypes, FileData, IdentifierData, ObjectData, RepositoryError, \
    assert_trailing_slash, clone_repository, get_full_path, get_latest_commit, read_file, \
    to_batches, transform_files_list

# TODO: better naming
# TODO: add AST functionality

# TODO: check the proper way to create a singleton
# One instance for further subtokenizing
subtokenizer = TokenParser()

# TODO: give the user the possibility the specify the number of processors
# Number of threads for multi-processing
PROCESSES = cpu_count()


def subtokenize_identifier(token: Union[str, IdentifierData]) -> \
        Union[List[str], List[IdentifierData]]:
    """
    Splits the identifier into subtokens.
    :param token: either a string of identifier or an IdentifierData object.
    :return: a list of the corresponding objects for each subtoken.
    """
    if isinstance(token, str):
        subtokens = [subtoken for subtoken in list(subtokenizer.process_token(token))]
    elif isinstance(token, IdentifierData):
        # Currently, each subtoken returns the coordinates of the original token.
        # TODO: fix the subtokenization to account for the change of coordinates.
        subtokens = [IdentifierData(identifier=subtoken, start_byte=token.start_byte,
                                    start_line=token.start_line,
                                    start_column=token.start_column)
                     for subtoken in list(subtokenizer.process_token(token.identifier))]
    else:
        raise TypeError("Unknown format of token!")
    return subtokens


# TODO: language names' normalization
# TODO: do we really need a class with only static methods?
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
        :param node: node of the AST.
        :return: (start byte, end byte).
        """
        start = node.start_byte
        end = node.end_byte
        return start, end

    @staticmethod
    def traverse_tree(node: tree_sitter.Node, types: Set[str]) -> Iterator[tree_sitter.Node]:
        """
        Run down the AST (DFS) from a given node and yield its children of necessary types.
        :param node: starting Tree-sitter node.
        :param types: the set of types of interest.
        :return: the iterator of Tree-sitter nodes of necessary types.
        """
        stack = deque([node])

        while stack:
            node = stack.popleft()
            stack.extendleft(reversed(node.children))
            if node.type in types:
                yield node

    @staticmethod
    def get_identifier_from_node(code: bytes, node: tree_sitter.Node,
                                 identifiers_verbose: bool = False) -> Union[str, IdentifierData]:
        """
        Given an identifier node of the AST and the code from which this AST was built,
        return the identifier.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :param identifiers_verbose: if True, will return not only the identifier itself,
                                    but also its parameters as IdentifierData.
        :return: str with just identifier or an IdentifierData object.
        """
        start_byte, end_byte = TreeSitterParser.get_positional_bytes(node)
        identifier = code[start_byte:end_byte].decode("utf-8")
        if not identifiers_verbose:
            return identifier
        else:
            start_line, start_column = node.start_point
            return IdentifierData(identifier, start_byte, start_line, start_column)

    @staticmethod
    def get_identifiers_sequence_from_node(code: bytes, node: tree_sitter.Node, lang: str,
                                           identifiers_verbose: bool = False,
                                           subtokenize: bool = False) -> \
            Union[List[str], List[IdentifierData]]:
        """
        Given a node of the AST and the code from which this AST was built, gather a list of
        identifiers in it.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :param lang: the language of code.
        :param identifiers_verbose: if True, will save not only identifiers themselves,
                                    but also their parameters as IdentifierData.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: list of identifiers as either strings or IdentifierData objects.
        """
        token_nodes = TreeSitterParser.traverse_tree(node, TreeSitterParser.IDENTIFIERS[lang])
        tokens_sequence = []

        for token_node in token_nodes:
            token = TreeSitterParser.get_identifier_from_node(code, token_node, identifiers_verbose)
            if not subtokenize:
                tokens_sequence.append(token)
            else:
                subtokens = subtokenize_identifier(token)
                tokens_sequence.extend(subtokens)

        return tokens_sequence

    @staticmethod
    def get_identifiers_sequence_from_code(code: str, lang: str, identifiers_verbose: bool = False,
                                           subtokenize: bool = False) -> \
            Union[List[str], List[IdentifierData]]:
        """
        Given the code and its language, gather identifiers in it.
        :param code: source code as a string.
        :param lang: language of the code.
        :param identifiers_verbose: if True, will save not only identifiers themselves,
                                    but also their parameters as IdentifierData.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: list of identifiers as either strings or IdentifierData objects.
        """
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        return TreeSitterParser.get_identifiers_sequence_from_node(code, root, lang,
                                                                   identifiers_verbose,
                                                                   subtokenize)

    @staticmethod
    def get_object_from_node(object_type: ObjectTypes, code: bytes, node: tree_sitter.Node,
                             lang: str, identifiers_verbose: bool = False,
                             subtokenize: bool = False) -> ObjectData:
        """
        Given a node of the AST, its type, and the code from which this AST was built,
        compile an ObjectData object.
        :param object_type: the type of the object.
        :param code: the original code in bytes.
        :param node: the node of the tree-sitter AST.
        :param lang: the language of code.
        :param identifiers_verbose: if True, will save not only identifiers themselves,
                                    but also their parameters as IdentifierData.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: ObjectData object.
        """
        start_byte, end_byte = TreeSitterParser.get_positional_bytes(node)
        content = code[start_byte:end_byte].decode("utf-8")
        start_line, start_column = node.start_point
        end_line, end_column = node.end_point
        identifiers = TreeSitterParser.get_identifiers_sequence_from_node(code, node, lang,
                                                                          identifiers_verbose,
                                                                          subtokenize)
        if identifiers_verbose:
            identifiers_type = IdentifiersTypes.VERBOSE
        else:
            identifiers_type = IdentifiersTypes.STRING
        return ObjectData(object_type=object_type, content=content, lang=lang,
                          identifiers=identifiers, identifiers_type=identifiers_type,
                          start_byte=start_byte, start_line=start_line, start_column=start_column,
                          end_byte=end_byte, end_line=end_line, end_column=end_column)

    @staticmethod
    def merge_nodes_for_lang(lang: str) -> Set[str]:
        """
        Given a language, get a set of all node types that are collected for it (identifiers,
        functions, and classes).
        :param lang: the language of code.
        :return: a set of tree-sitter node types.
        """
        identifier_types, function_types, class_types = set(), set(), set()
        if lang in TreeSitterParser.IDENTIFIERS.keys():
            identifier_types = TreeSitterParser.IDENTIFIERS[lang]
        if lang in TreeSitterParser.FUNCTIONS.keys():
            function_types = TreeSitterParser.FUNCTIONS[lang]
        if lang in TreeSitterParser.CLASSES.keys():
            class_types = TreeSitterParser.CLASSES[lang]
        return identifier_types | function_types | class_types

    # TODO: check pipeline patterns, refactor
    @staticmethod
    def get_data_from_file(file: str, lang: str, gather_objects: bool, gather_identifiers: bool,
                           identifiers_verbose: bool = False,
                           subtokenize: bool = False) -> FileData:
        """
        Given a file and its language, return a FileData object.
        :param file: the path to file.
        :param lang: the language of code.
        :param gather_objects: if True, will gather ObjectData objects for classes and functions.
        :param gather_identifiers: if True, will gather a list of identifiers for the file.
        :param identifiers_verbose: if True, will save not only identifiers themselves,
                                    but also their parameters as IdentifierData.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: FileData object.
        """
        code = read_file(file)
        code = bytes(code, "utf-8")
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(code)
        root = tree.root_node
        if identifiers_verbose:
            identifiers_type = IdentifiersTypes.VERBOSE
        else:
            identifiers_type = IdentifiersTypes.STRING

        identifiers = []
        objects = []

        # The tree is traversed once per file
        for node in TreeSitterParser.traverse_tree(
                root, TreeSitterParser.merge_nodes_for_lang(lang)):
            # Gathering identifiers for the file
            if node.type in TreeSitterParser.IDENTIFIERS[lang]:
                if gather_identifiers:
                    if not subtokenize:
                        identifiers.append(TreeSitterParser
                                           .get_identifier_from_node(code, node,
                                                                     identifiers_verbose))
                    else:
                        identifiers.extend(subtokenize_identifier(
                            TreeSitterParser.get_identifier_from_node(code, node,
                                                                      identifiers_verbose)))
            # Gathering ObjectData for functions
            elif node.type in TreeSitterParser.FUNCTIONS[lang]:
                if gather_objects:
                    objects.append(TreeSitterParser.get_object_from_node(ObjectTypes.FUNCTION,
                                                                         code, node, lang,
                                                                         identifiers_verbose,
                                                                         subtokenize))
            # Gathering ObjectData for classes
            elif node.type in TreeSitterParser.CLASSES[lang]:
                if gather_objects:
                    objects.append(TreeSitterParser.get_object_from_node(ObjectTypes.CLASS, code,
                                                                         node, lang,
                                                                         identifiers_verbose,
                                                                         subtokenize))

        return FileData(path=file, lang=lang, objects=objects, identifiers=identifiers,
                        identifiers_type=identifiers_type)


# TODO: common parent class for Pygments and Tree-sitter
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
    def get_identifiers_sequence_from_code(code: str, lang: str, identifiers_verbose: bool = False,
                                           subtokenize: bool = False) -> \
            Union[List[str], List[IdentifierData]]:
        """
        Given the code and its language, gather its identifiers.
        :param code: the code to parse.
        :param lang: the language of the code fragment.
        :param identifiers_verbose: if True, will save not only identifiers themselves,
                                    but also their parameters as IdentifierData.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: list of identifiers as either strings or IdentifierData objects.
        """
        tokens = []
        for pair in pygments.lex(code, PygmentsParser.LEXERS[lang]):
            if any(pair[0] in sublist for sublist in PygmentsParser.IDENTIFIERS[lang]):
                # TODO: implement indexes for tokens, it's possible in pygments. (0, 0, 0) for now.
                if not identifiers_verbose:
                    token = pair[1]
                else:
                    token = IdentifierData(pair[1], 0, 0, 0)
                if not subtokenize:
                    tokens.append(token)
                else:
                    tokens.extend(subtokenize_identifier(token))
        return tokens

    @staticmethod
    def get_data_from_file(file: str, lang: str, identifiers_verbose: bool = False,
                           subtokenize: bool = False) -> FileData:
        """
        Given a file and its language, return a FileData object.
        :param file: path to file.
        :param lang: the language of code.
        :param identifiers_verbose: if True, will save not only identifiers themselves,
                                    but also their parameters as IdentifierData.
        :param subtokenize: if True, will split the tokens into subtokens.
        :return: FileData object.
        """
        code = read_file(file)
        identifiers = PygmentsParser.get_identifiers_sequence_from_code(code, lang,
                                                                        identifiers_verbose,
                                                                        subtokenize)
        if identifiers_verbose:
            identifiers_type = IdentifiersTypes.VERBOSE
        else:
            identifiers_type = IdentifiersTypes.STRING
        # The "objects" are always empty, because Pygments don't support recognizing them.
        return FileData(path=file, lang=lang, objects=[], identifiers=identifiers,
                        identifiers_type=identifiers_type)


def get_identifiers_sequence_from_code(code: str, lang: str, identifiers_verbose: bool = False,
                                       subtokenize: bool = False) -> \
        Union[List[str], List[IdentifierData]]:
    """
    Given the code and its language, gather its identifiers.
    :param code: the code to parse.
    :param lang: the language of the code fragment.
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: list of identifiers as either strings or IdentifierData objects.
    """
    if lang in SUPPORTED_LANGUAGES["tree-sitter"]:
        return TreeSitterParser.get_identifiers_sequence_from_code(code, lang, identifiers_verbose,
                                                                   subtokenize)
    elif lang in SUPPORTED_LANGUAGES["pygments"]:
        return PygmentsParser.get_identifiers_sequence_from_code(code, lang, identifiers_verbose,
                                                                 subtokenize)
    else:
        raise ValueError("Unsupported language!")


def get_identifiers_sequence_from_file(file: str, lang: str, identifiers_verbose: bool = False,
                                       subtokenize: bool = False) -> \
        Union[List[str], List[IdentifierData]]:
    """
    Given the file and its language, gather subtokens of identifiers as IdentifierData objects.
    :param file: path to file.
    :param lang: the language of code.
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: list of identifiers as either strings or IdentifierData objects.
    """
    code = read_file(file)
    return get_identifiers_sequence_from_code(code, lang, identifiers_verbose, subtokenize)


def get_data_from_file(file: str, lang: str, gather_objects: bool, gather_identifiers: bool,
                       identifiers_verbose: bool = False,
                       subtokenize: bool = False) -> FileData:
    """
    Given a file and its language, return a FileData object.
    :param file: path to file.
    :param lang: the language of code.
    :param gather_objects: if True, will gather ObjectData objects for classes and functions.
    :param gather_identifiers: if True, will gather a list of identifiers for the file.
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: FileData object.
    """
    logging.debug(f"Getting FileData from {file}.")
    try:
        if lang in SUPPORTED_LANGUAGES["tree-sitter"]:
            return TreeSitterParser.get_data_from_file(file, lang, gather_objects,
                                                       gather_identifiers, identifiers_verbose,
                                                       subtokenize)
        elif lang in SUPPORTED_LANGUAGES["pygments"]:
            return PygmentsParser.get_data_from_file(file, lang, identifiers_verbose, subtokenize)
        else:
            raise ValueError("Unsupported language!")
    except UnicodeDecodeError:
        logging.warning(f"UnicodeDecodeError in {file}, skipping...")
        # Returning empty file for multiprocessing and further skipping during saving.
        return FileData(path=file, lang=lang, objects=[], identifiers=[],
                        identifiers_type=IdentifiersTypes.STRING)


def get_functions_from_file(file: str, lang: str, identifiers_verbose: bool = False,
                            subtokenize: bool = False) -> List[ObjectData]:
    """
    Yield ObjectData objects for functions in a given file.
    :param file: the path to file.
    :param lang: the language of the file.
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: an iterator of ObjectData objects for functions.
    """
    if lang not in SUPPORTED_LANGUAGES["functions"]:
        raise ValueError(f"{lang} doesn't support gathering functions!")
    file_data = TreeSitterParser.get_data_from_file(file, lang, gather_objects=True,
                                                    gather_identifiers=False,
                                                    identifiers_verbose=identifiers_verbose,
                                                    subtokenize=subtokenize)
    for obj in file_data.objects:
        if obj.object_type == ObjectTypes.FUNCTION:
            yield obj


def get_classes_from_file(file: str, lang: str, identifiers_verbose: bool = False,
                          subtokenize: bool = False) -> List[ObjectData]:
    """
    Yield ObjectData objects for classes in a given file.
    :param file: the path to file.
    :param lang: the language of the file.
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: an iterator of ObjectData objects for classes.
    """
    if lang not in SUPPORTED_LANGUAGES["classes"]:
        raise ValueError(f"{lang} doesn't support gathering functions!")
    file_data = TreeSitterParser.get_data_from_file(file, lang, gather_objects=True,
                                                    gather_identifiers=False,
                                                    identifiers_verbose=identifiers_verbose,
                                                    subtokenize=subtokenize)
    for obj in file_data.objects:
        if obj.object_type == ObjectTypes.CLASS:
            yield obj


# TODO: functionality for GitHub link creation
def tokenize_repository(repository: str, local: bool, mode: str, gran: str,
                        languages: Optional[List[str]], pool: Parallel,
                        identifiers_verbose: bool = False,
                        subtokenize: bool = False) -> Tuple[str, List[FileData]]:
    """
    Tokenize a given repository. Return its correct full name and a list of FileData objects.
    :param repository: a link to the repository. If "local" is False, a link to GitHub,
                       otherwise - a path to a directory.
    :param local: True if tokenizing in local mode (repository is a path to a directory),
                  False if tokenizing in default mode (repository is a GitHub link).
    :param mode: the mode of parsing. Either "counters" or "sequences".
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param languages: the languages of parsing. None for all the languages available for a
                      given parsing granularity, specific languages for themselves.
    :param pool: the Parallel class instance for multiprocessing.
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: the correct name of the repository for links and a list of FileData objects.
    """
    repository = assert_trailing_slash(repository)
    with TemporaryDirectory() as td:
        # Determine the correct working directory and its name
        if local:
            directory = repository  # Working directly with a path in the local mode
            if not os.path.isdir(directory):
                raise RepositoryError(f"{directory} isn't a directory!")
            repository_name = directory
        else:
            logging.debug(f"Cloning {repository}.")
            directory = td  # Working with a temporary directory in the remote mode
            clone_repository(repository, directory)  # Cloning the repository
            # The name of the repository includes the commit for working links.
            commit = get_latest_commit(directory)
            repository_name = f"{repository}tree/{commit}/"
        logging.debug(f"Recognizing languages is {repository}.")
        lang2files = recognize_languages_dir(directory)  # Recognize the languages in the directory
        files = transform_files_list(lang2files, gran, languages)
        logging.debug(f"Parsing files in {repository}.")
        # Parsing for files and projects does not require gathering ObjectData objects.
        # TODO: avoid hardcoded names
        if gran in ["projects", "files"]:
            gather_objects = False
            gather_identifiers = True
        # Parsing for classes and functions does not require gathering identifiers for files.
        else:
            gather_objects = True
            gather_identifiers = False
        # Full parameters of identifiers can't be saved for counters.
        if (mode == "counters") and (identifiers_verbose is True):
            logging.warning("Full parameters of identifiers can't be saved in 'counters' mode!")
            identifiers_verbose = False
        chunk_results = pool([delayed(get_data_from_file)
                              (get_full_path(file[0], directory), file[1], gather_objects,
                               gather_identifiers, identifiers_verbose, subtokenize)
                              for file in files])
        logging.debug(f"Gathering results for {repository}.")
        files = []
        for file in chunk_results:
            # In the remote mode, the temporary directory is changed for the GitHub link.
            if not local:
                file.path = repository_name + os.path.relpath(file.path, directory)
            files.append(file)
    return repository_name, files


def tokenize_list_of_repositories(repositories_file: str, output_dir: str, batch_size: int,
                                  mode: str, gran: str, languages: Optional[List[str]],
                                  local: bool, output_format: str,
                                  identifiers_verbose: bool = False,
                                  subtokenize: bool = False) -> None:
    """
    Given the list of links to repositories, tokenize all the repositories in the list,
    writing them in batches to files in a specified output format.
    :param repositories_file: path to text file with a list of repositories.
    :param output_dir: path to the output directory.
    :param batch_size: the number of repositories to be grouped into a single batch / file.
    :param gran: granularity of parsing. Values are ["projects", "files", "classes", "functions"].
    :param mode: the mode of parsing. Either "counters" or "sequences".
    :param languages: the languages of parsing. None for all the languages available for a
                      given parsing granularity, specific languages for themselves.
    :param local: True if tokenizing in local mode (the input file contains paths to directories),
                  False if tokenizing in default mode (the input file contains GitHub links).
    :param output_format: the output format. Possible values: ["wabbit", "json"].
    :param identifiers_verbose: if True, will save not only identifiers themselves,
                                but also their parameters as IdentifierData.
    :param subtokenize: if True, will split the tokens into subtokens.
    :return: None.
    """
    if gran not in GRANULARITIES:
        raise ValueError("Incorrect granularity of parsing.")
    if mode not in PARSING_MODES:
        raise ValueError("Incorrect parsing mode.")
    if output_format not in OUTPUT_FORMATS:
        raise ValueError("Incorrect output format.")
    logging.info(f"Tokenizing the repositories in {mode} mode, with {gran} granularity, "
                 f"saving into {output_format} format. Specific languages: {languages}, "
                 f"subtokenizing: {subtokenize}, "
                 f"parameters of identifiers: {identifiers_verbose}.")
    # Reading the input file and splitting repositories into batches.
    with open(repositories_file) as fin:
        repositories_list = fin.read().splitlines()
        repositories_batches = to_batches(repositories_list, batch_size)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # TODO: user should configure the number of processes
    with Parallel(PROCESSES) as pool:
        # Iterating over batches
        for count_batch, batch in enumerate(repositories_batches):
            logging.info(f"Tokenizing batch {count_batch + 1} out of {len(repositories_batches)}.")
            reps2files = {}
            filename = f"{output_format}_{mode}_{gran}_{count_batch}.txt"
            # Iterating over repositories in the batch
            # TODO: add progress bar
            for count_repository, repository in enumerate(batch):
                logging.info(f"Tokenizing repository: {repository} ({count_repository + 1} "
                             f"out of {len(batch)} in batch {count_batch + 1}).")
                try:
                    repository_name, files = tokenize_repository(repository, local, mode,
                                                                 gran, languages, pool,
                                                                 identifiers_verbose,
                                                                 subtokenize)
                except RepositoryError:
                    logging.warning(f"{repository} is an incorrect link, skipping...")
                    continue
                reps2files[repository_name] = files
            logging.info(f"Writing batch {count_batch + 1} out "
                         f"of {len(repositories_batches)} to file.")
            if len(reps2files.keys()) != 0:  # Skipping possible empty batches.
                # Saving the batch in the necessary format.
                OutputFormats(output_format, reps2files, mode, gran, output_dir, filename)
            logging.info(f"Finished batch {count_batch + 1} out of {len(repositories_batches)}.")
    logging.info("Tokenization successfully completed.")
