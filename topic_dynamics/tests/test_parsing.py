"""
Parsing-related tests.
"""

import os
from typing import List, Tuple
import unittest

from ..parsing import get_identifiers

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestParser(unittest.TestCase):

    test_parser_data = [["java", "test_files/test.java",
                         [("i", 9), ("anarray", 6), ("length", 2), ("system", 2), ("out", 2), ("arraydemo", 1),
                          ("main", 1), ("string", 1), ("args", 1), ("print", 1), ("println", 1)]],
                        ["python", "test_files/test.py",
                         [("left", 4), ("right", 4), ("n", 4), ("board_size", 3), ("col", 3), ("solve", 3),
                          ("solution", 3), ("i", 3), ("under_attack", 2), ("queens", 2), ("c", 2),
                          ("smaller_solutions", 2),
                          ("answer", 2), ("r", 1), ("reversed", 1), ("range", 1), ("print", 1)]],
                        ["cpp", "test_files/test.cpp",
                         [("n", 4), ("arr", 2), ("std", 1), ("main", 1), ("cout", 1), ("endl", 1)]]]

    def test_parser(self):
        for data in TestParser.test_parser_data:
            with self.subTest():
                file = os.path.abspath(os.path.join(tests_dir, data[1]))
                identifiers = get_identifiers(file, data[0])
                self.assertEqual(identifiers, data[2])


if __name__ == "__main__":
    unittest.main()
