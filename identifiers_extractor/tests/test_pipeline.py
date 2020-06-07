"""
Pipeline-related tests.
"""
from collections import Counter
import os
import unittest

from ..parsing import cmdline, recognize_languages, transform_files_list, transform_tokens, \
    tokenize_list_of_repositories

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestPipeline(unittest.TestCase):
    def test_cmdline(self):
        command = "echo 'Darina'"
        stdout = cmdline(command)
        self.assertEqual(stdout, "Darina")

    def test_languages(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(lang2files), 16)
        self.assertEqual(lang2files.keys(),
                         {"C", "C#", "C++", "Go", "Haskell", "Java", "JavaScript", "Kotlin", "PHP",
                          "Python", "Ruby", "Rust", "Scala", "Shell", "Swift", "TypeScript"})

    def test_transforming_list(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        files = transform_files_list(lang2files, "projects")
        self.assertEqual(len(files), 16)

    def test_transforming_tokens(self):
        tokens = Counter({"height": 4, "width": 4, "rectangl": 2,
                          "calc": 2, "area": 2, "prototyp": 1})
        transformed_tokens = transform_tokens(tokens)
        self.assertEqual(transformed_tokens, ["area:2", "calc:2", "height:4",
                                              "prototyp:1", "rectangl:2", "width:4"])

    def test_tokenization(self):
        tokenize_list_of_repositories(os.path.abspath(os.path.join(
            tests_dir, "test_files", "test.txt")), os.path.abspath(
            os.path.join(tests_dir, "test_results")), 100, "files", True, "wabbit")
        with open(os.path.abspath(os.path.join(tests_dir, "test_results",
                                               "wabbit_files_0.txt"))) as fin:
            wabbit_lines = sum(1 for line in fin)
        self.assertEqual(wabbit_lines, 16)


if __name__ == "__main__":
    unittest.main()
