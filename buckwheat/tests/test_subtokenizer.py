"""
Subtokenizing-related tests.
"""
import os
import unittest

from ..subtokenizer import TokenParser

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestSubtokenizing(unittest.TestCase):
    test_subtokenizing_data = [["token", ["token"]],
                               ["Upper", ["upper"]],
                               ["camelCase", ["camel", "case"]],
                               ["snake_case", ["snake", "case"]],
                               ["os", []],
                               ["wdSize", ["size", "wdsize"]],
                               ["Egor.is.Nice", ["egor", "nice", "isnice"]],
                               ["stemming", ["stem"]],
                               ["sourced_directory", ["sourc", "directori"]],
                               ["some.ABSUrdSpecific_case.ml.in.code",
                                ["some", "abs", "urd", "specif", "case", "code", "incode"]]]

    Subtokenizer = TokenParser()

    def test_subtokenizer(self):
        for data in TestSubtokenizing.test_subtokenizing_data:
            with self.subTest():
                subtokens = list(TestSubtokenizing.Subtokenizer.process_token(data[0]))
                self.assertEqual(subtokens, data[1])


if __name__ == "__main__":
    unittest.main()
