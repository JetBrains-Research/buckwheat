"""
Pipeline-related tests.
"""
from collections import Counter
import os
import unittest

from ..parsing import cmdline, recognize_languages, transform_files_list, transform_tokens, \
    tokenize_repositories

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestPipeline(unittest.TestCase):
    def test_cmdline(self):
        command = "echo 'Darina'"
        stdout = cmdline(command)
        self.assertEqual(stdout, "Darina\n")

    def test_languages(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(lang2files), 16)
        self.assertEqual(lang2files.keys(),
                         {"C", "C#", "C++", "Go", "Haskell", "Java", "JavaScript", "Kotlin", "PHP",
                          "Python", "Ruby", "Rust", "Scala", "Shell", "Swift", "TypeScript"})

    def test_transforming_list(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        files = transform_files_list(lang2files,
                                     os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(files), 16)

    def test_transforming_tokens(self):
        tokens = Counter({"height": 4, "width": 4, "rectangl": 2,
                          "calc": 2, "area": 2, "prototyp": 1})
        token2number = {"height": 228, "width": 11, "rectangl": 6,
                          "calc": 2, "area": 322, "prototyp": 25}
        transformed_tokens = transform_tokens(tokens, token2number)
        self.assertEqual(transformed_tokens, ["228:4", "11:4", "6:2", "2:2", "322:2", "25:1"])

    def test_tokenization(self):
        tokenize_repositories(os.path.abspath(os.path.join(tests_dir, "test_files", "test.txt")),
                              os.path.abspath(os.path.join(tests_dir, "test_results")),
                              100, True)
        with open(os.path.abspath(os.path.join(tests_dir, "test_results", "vocab0.txt"))) as fin:
            vocab_lines = sum(1 for line in fin)
        self.assertEqual(vocab_lines, 221)


if __name__ == "__main__":
    unittest.main()
