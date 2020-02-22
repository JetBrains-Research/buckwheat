"""
Parsing-related tests.
"""

import os
import unittest

from ..parsing import get_identifiers

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestParser (unittest.TestCase):
    def test_java(self):
        # Test parsing a Java file.
        file = os.path.join(tests_dir, "test_files/test.java")
        identifiers = get_identifiers(file, "java")
        real_identifiers = [('i', 9), ('anArray', 6), ('length', 2), ('System', 2), ('out', 2), ('ArrayDemo', 1), ('main', 1), ('String', 1), ('args', 1), ('print', 1), ('println', 1)]
        self.assertEqual(identifiers, real_identifiers)

    def test_python(self):
        # Test parsing a Python file.
        file = os.path.join(tests_dir, "test_files/test.py")
        identifiers = get_identifiers(file, "python")
        real_identifiers = [('left', 4), ('right', 4), ('n', 4), ('BOARD_SIZE', 3), ('col', 3), ('solve', 3), ('solution', 3), ('i', 3), ('under_attack', 2), ('queens', 2), ('c', 2), ('smaller_solutions', 2), ('answer', 2), ('r', 1), ('reversed', 1), ('range', 1), ('print', 1)]
        self.assertEqual(identifiers, real_identifiers)

    def test_cpp(self):
        # Test parsing a C++ file.
        file = os.path.join(tests_dir, "test_files/test.cpp")
        identifiers = get_identifiers(file, "cpp")
        real_identifiers = [('n', 4), ('arr', 2), ('std', 1), ('main', 1), ('cout', 1), ('endl', 1)]
        self.assertEqual(identifiers, real_identifiers)


if __name__ == '__main__':
    unittest.main()
