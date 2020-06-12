"""
Pipeline-related tests.
"""
import os
import unittest

from ..main import recognize_languages, tokenize_list_of_repositories, transform_files_list

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestPipeline(unittest.TestCase):

    def test_languages(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(lang2files), 16)
        self.assertEqual(lang2files.keys(),
                         {"C", "C#", "C++", "Go", "Haskell", "Java", "JavaScript", "Kotlin", "PHP",
                          "Python", "Ruby", "Rust", "Scala", "Shell", "Swift", "TypeScript"})

    def test_transforming_list(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        files = transform_files_list(lang2files, "projects", "all")
        self.assertEqual(len(files), 16)

    def test_tokenization(self):
        tokenize_list_of_repositories(os.path.abspath(os.path.join(
            tests_dir, "test_files", "test.txt")), os.path.abspath(
            os.path.join(tests_dir, "test_results")),
            100, "counters", "files", "all", True, "wabbit")
        with open(os.path.abspath(os.path.join(tests_dir, "test_results",
                                               "wabbit_counters_files_0.txt"))) as fin:
            wabbit_lines = sum(1 for _ in fin)
        self.assertEqual(wabbit_lines, 16)


if __name__ == "__main__":
    unittest.main()
