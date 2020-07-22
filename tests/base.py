from pathlib import Path
from typing import List

from buckwheat.pipeline.input import classify_languages_in_directory, LanguageClassifiedFile, \
    LanguageClassifiedDirectory, transform_directory_to_files

tests_files_dir = Path(__file__).parent / "samples" / "identifiers_extraction"


def get_directory_instance() -> LanguageClassifiedDirectory:
    return classify_languages_in_directory(tests_files_dir)


def get_classified_test_files() -> List[LanguageClassifiedFile]:
    return transform_directory_to_files(get_directory_instance())


def get_test_files_list() -> List[str]:
    return list(map(str, tests_files_dir.glob("*")))
