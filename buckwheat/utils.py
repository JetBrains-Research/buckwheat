"""
Auxiliary functionality.
"""
import dataclasses
from enum import Enum
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple, Union

# TODO: better naming

# Languages supported in various
# TODO: language names' normalization (c_sharp, C#, csharp -> C#), maybe within Enry?
SUPPORTED_LANGUAGES = {"tree-sitter": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                       "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"},
                       "pygments": {"Scala", "Swift", "Kotlin", "Haskell"},
                       "classes": {"JavaScript", "Python", "Java", "C++", "Ruby", "TypeScript",
                                   "TSX", "PHP", "C#"},
                       "functions": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                     "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust"},
                       "comments": {"JavaScript", "Python", "Java", "Go", "C++", "Ruby",
                                    "TypeScript", "TSX", "PHP", "C#", "C", "Shell", "Rust",
                                    "Scala", "Swift", "Kotlin", "Haskell"}}

# Supported parsing modes
PARSING_MODES = {"sequences", "counters"}

# Supported granularities of parsing
GRANULARITIES = {"projects", "files", "classes", "functions"}

# Supported output formats
OUTPUT_FORMATS = {"wabbit", "json"}


class ObjectTypes(Enum):
    CLASS = "class"
    FUNCTION = "function"


class IdentifiersTypes(Enum):
    """
    string = identifier itself
    verbose = IdentifierData class
    """
    STRING = "string"
    VERBOSE = "verbose"


# TODO: consider the differences between str and byte from the standpoint of coordinates
@dataclasses.dataclass
class IdentifierData:
    """
    Data class to store individual identifiers and their positional coordinates.
    """
    identifier: str
    start_byte: int
    start_line: int
    start_column: int


@dataclasses.dataclass
class ObjectData:
    """
    Data class to store objects (classes and functions) and their parameters: positional
    coordinates, language and identifiers.
    """
    object_type: ObjectTypes
    content: str
    lang: str
    identifiers: Union[List[IdentifierData], List[str]]
    identifiers_type: IdentifiersTypes  # VERBOSE for IdentifierData, STRING for str.
    start_byte: int
    start_line: int
    start_column: int
    end_byte: int
    end_line: int
    end_column: int


# TODO: think about the duplication of identifiers_type
@dataclasses.dataclass
class FileData:
    """
    Dataclass to store files and their content.
    """
    path: str
    lang: str
    objects: List[ObjectData]
    identifiers: Union[List[IdentifierData], List[str]]
    identifiers_type: IdentifiersTypes  # VERBOSE for IdentifierData, STRING for str.


class RepositoryError(ValueError):
    """
    A special error for catching wrong links to repositories and skipping such repositories.
    """
    pass


def read_file(file: str) -> str:
    """
    Read the contents of the file.
    :param file: the path to the file.
    :return: the contents of the file.
    """
    with open(file) as fin:
        return fin.read()


def to_batches(lst: List[Any], batch_size: int) -> List[List[Any]]:
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


# TODO: maybe do it without modifying the link (consider GIT_TERMINAL_PROMT=0)
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
    command = f"cd {directory}; git rev-parse HEAD"
    return subprocess.check_output(command, shell=True, text=True).rstrip()


def get_full_path(file: str, directory: str) -> str:
    """
    Get the full path to file from the full path to a directory and a relative path to that
    file in that directory.
    :param file: the relative path to file in a directory.
    :param directory: the full path of a directory.
    :return: the full path to file.
    """
    return os.path.abspath(os.path.join(directory, file))


# TODO: Avoid hardcoded values
def transform_files_list(lang2files: Dict[str, List[str]], gran: str,
                         languages: Optional[List[str]]) -> List[Tuple[str, str]]:
    """
    Transform the output of Enry on a directory into a list of tuples (full_path_to_file, lang)
    for supported languages only. Supported languages depend on the granularity and whether one
    specific language was specified.
    :param lang2files: the dictionary output of Enry: {language: [files], ...}.
    :param gran: granularity of parsing. Values are in GRANULARITIES const.
    :param languages: the languages of parsing. None for all the languages available for a
                      given parsing granularity, specific languages for themselves.
    :return: a list of tuples (full_path_to_file, lang) for the necessary languages.
    """
    # Get the languages available for a given granularity.
    if gran in ["projects", "files"]:  # Projects and files are supported for all languages.
        res_langs = SUPPORTED_LANGUAGES["tree-sitter"] | SUPPORTED_LANGUAGES["pygments"]
    elif gran == "classes":
        res_langs = SUPPORTED_LANGUAGES["classes"]
    elif gran == "functions":
        res_langs = SUPPORTED_LANGUAGES["functions"]
    else:
        # TODO: Better error messages
        raise ValueError("Incorrect granularity of parsing.")
    # If specific languages were specified, override the results
    # and check their availability for a given granularity.
    if languages is not None:
        for language in languages:
            if language not in SUPPORTED_LANGUAGES["tree-sitter"] | \
                    SUPPORTED_LANGUAGES["pygments"]:
                raise ValueError(f"{language} is an unsupported language!")
            if language not in res_langs:
                raise ValueError(f"{language} doesn't support {gran} granularity.")
        res_langs = set(languages)
    files = []
    for lang in lang2files:
        if lang in res_langs:
            for file in lang2files[lang]:
                files.append((file, lang))
    return files
