import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

from buckwheat import recognize_languages_dir
from buckwheat.utils import clone_repository, ProgrammingLanguages


@dataclass
class LanguageClassifiedDirectory:
    """Directory with list of classified files"""
    path: str
    language_file_index: Dict[str, List[str]]

    def get_language_of_file(self, file_path: str) -> Optional[str]:
        for lang, files in self.language_file_index.items():
            if file_path in files:
                return lang


@dataclass
class LanguageClassifiedGitDirectory(LanguageClassifiedDirectory):
    repository: str


@dataclass
class LanguageClassifiedFile:
    """File with classified language"""
    path: str
    language: ProgrammingLanguages


def get_repositories_list(input_file: str) -> List[str]:
    """
    Return list of repositories list from input file

    :param input_file: file with list of repositories specified
    :return: list of repository links from file
    """
    with open(input_file, "r") as repositories_file_list:
        return repositories_file_list.readlines()


def clone_repository_to_temp_dir(repository_link: str) -> str:
    """
    Clone repository to temp_dir and return it's path

    :param repository_link:
    :return:
    """
    temp_directory = tempfile.mkdtemp()
    clone_repository(repository_link, temp_directory)
    return temp_directory


def classify_languages_in_directory(repository_dir: str) -> LanguageClassifiedDirectory:
    """
    Construct repository instance and create all needed folders for parsing

    :param repository_dir: dir containing repository
    :return: repository instance
    """
    return LanguageClassifiedDirectory(repository_dir, recognize_languages_dir(repository_dir))


def transform_directory_to_files(directory: LanguageClassifiedDirectory) -> List[LanguageClassifiedFile]:
    """
    Transform directory to list of files with programming language specified

    :param directory: repository instance
    :return: list of files with programming languages
    """
    return [
        LanguageClassifiedFile(os.path.join(directory.path, file), ProgrammingLanguages(lang))
        for lang, files in directory.language_file_index.items()
        for file in files
    ]
