"""
Pipeline-related tests.
"""
import os
from tempfile import TemporaryDirectory
import unittest

from ..tokenizer import recognize_languages_dir, tokenize_list_of_repositories, transform_files_list

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestPipeline(unittest.TestCase):

    def test_languages(self):
        lang2files = recognize_languages_dir(os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(lang2files), 16)
        self.assertEqual(lang2files.keys(),
                         {"C", "C#", "C++", "Go", "Haskell", "Java", "JavaScript", "Kotlin", "PHP",
                          "Python", "Ruby", "Rust", "Scala", "Shell", "Swift", "TypeScript"})

    def test_transforming_list(self):
        lang2files = recognize_languages_dir(os.path.abspath(os.path.join(tests_dir, "test_files")))
        files = transform_files_list(lang2files, "projects", None)
        self.assertEqual(len(files), 16)

    def test_tokenization(self):
        with TemporaryDirectory() as td:
            tokenize_list_of_repositories(os.path.abspath(os.path.join(
                tests_dir, "test_files", "test.txt")), td, 100, "counters", "files", None, True,
                "wabbit", identifiers_verbose=False, subtokenize=True)
            with open(os.path.abspath(os.path.join(td, "wabbit_counters_files_0.txt"))) as fin:
                wabbit_lines = sum(1 for _ in fin)
        self.assertEqual(wabbit_lines, 16)


if __name__ == "__main__":
    unittest.main()
