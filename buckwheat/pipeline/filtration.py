from dataclasses import dataclass
from itertools import chain
from typing import List, Generator, Tuple, Dict, Callable

from git import Repo

from buckwheat.pipeline.input import LanguageClassifiedDirectory, LanguageClassifiedFile, LanguageClassifiedGitDirectory
from buckwheat.utils import ProgrammingLanguages


class BaseDiffSource:

    def changed_file_paths(self) -> List[str]:
        raise NotImplementedError


@dataclass
class GitDiffSource(BaseDiffSource):
    directory: LanguageClassifiedDirectory
    start_rev: str

    def __post_init__(self):
        self.repository = Repo(self.directory.path)

    @property
    def changed_files(self) -> Generator[LanguageClassifiedFile, None, None]:
        head_commit = self.repository.commit("HEAD")
        target_commit = self.repository.commit(self.start_rev)

        diff = head_commit.diff(target_commit)

        # Iterate over added or modified files
        for file in chain(diff.iter_change_type("M"), diff.iter_change_type("A")):
            changed_file_path = file.b_path
            file_lang = self.directory.get_language_of_file(changed_file_path)

            if file_lang:
                yield LanguageClassifiedFile(changed_file_path, ProgrammingLanguages(file_lang))


GitFilesExtractor = Callable[[LanguageClassifiedGitDirectory], Tuple[LanguageClassifiedFile, ...]]


def build_git_modified_files_extractor(rev_spec: Dict[str, str]) -> GitFilesExtractor:
    def files_extractor(directory: LanguageClassifiedGitDirectory) -> Tuple[LanguageClassifiedFile, ...]:
        start_rev = rev_spec.get(directory.repository)
        # TODO: Return all files if no start rev is provided
        if not start_rev:
            return tuple()
        return tuple(GitDiffSource(directory, start_rev).changed_files)

    return files_extractor
