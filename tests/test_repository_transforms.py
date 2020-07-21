from tests.base import get_directory_instance
from buckwheat.pipeline.input import transform_directory_to_files

FILES_COUNT = 16
CLASSIFIED_FILES = {'C': ['test.c'], 'C#': ['test.cs'], 'C++': ['test.cpp'], 'Go': ['test.go'], 'Haskell': ['test.hs'],
                    'Java': ['test.java'], 'JavaScript': ['test.js'], 'Kotlin': ['test.kt'], 'PHP': ['test.php'],
                    'Python': ['test.py'], 'Ruby': ['test.rb'], 'Rust': ['test.rs'], 'Scala': ['test.scala'],
                    'Shell': ['test.sh'], 'Swift': ['test.swift'], 'TypeScript': ['test.ts']}


def test_language_classification():
    classified_directory = get_directory_instance()
    assert len(classified_directory.language_file_index.items()) == 16
    assert classified_directory.language_file_index == CLASSIFIED_FILES


def test_transform_repository_to_files():
    classified_directory = get_directory_instance()
    language_files = transform_directory_to_files(classified_directory)
    languages_set = set(file.language.value for file in language_files)
    assert len(language_files) == 16
    assert languages_set == CLASSIFIED_FILES.keys()
