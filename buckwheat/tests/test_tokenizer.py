"""
Parsing-related tests.
"""
from collections import Counter
import os
import unittest

from ..tokenizer import get_identifiers_sequence_from_file, get_comments_from_file

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestParser(unittest.TestCase):
    test_parser_data = [["JavaScript", "test.js",
                         {"height": 4, "width": 4, "rectangl": 2,
                          "calc": 2, "area": 2, "prototyp": 1},
                         ["// Rectangle instance declaration",
                          "/* Calculating area of Rectangle.\nCreate"
                          " function clacArea.\nArea calculations"
                          " = height * width.\n */"]],
                        ["Python", "test.py",
                         {"tree": 6, "sitter": 6, "dir": 4, "get": 3, "path": 3, "parser": 2,
                          "str": 2, "bin": 2, "loc": 2, "languag": 1, "abspath": 1, "dirnam": 1,
                          "file": 1, "join": 1},
                         ["# No parsers need to be declared here"]],
                        ["Java", "test.java",
                         {"stats": 7, "sub": 7, "dir": 4, "writer": 3, "file": 3, "dirs": 3,
                          "now": 3, "statist": 3, "all": 3, "collect": 2, "write": 2, "list": 2,
                          "featur": 2, "total": 2, "holder": 2, "and": 1, "print": 1, "string": 1,
                          "except": 1, "ioexcept": 1, "length": 1, "directori": 1,
                          "isdirectori": 1, "system": 1, "out": 1, "println": 1, "get": 1,
                          "name": 1, "collector": 1, "from": 1, "project": 1, "path": 1,
                          "topath": 1, "size": 1, "min": 1, "files": 1, "test": 1, "totest": 1},
                         ["/* Function that collects and writes statistics.subDirs\n"
                          "    Uses StatisticsHolder class for statistics information.\n    */",
                          "// total number of subDirs"]],
                        ["Go", "test.go",
                         {"buff": 4, "file": 4, "fmt": 3, "fprintln": 3, "err": 3, "out": 2,
                          "string": 2, "name": 2, "languag": 2, "version": 1, "commit": 1,
                          "print": 1, "break": 1, "down": 1, "buffer": 1, "data": 1, "read": 1,
                          "limit": 1},
                         ["// Declaration of variables", "/* Breaking Down function\n   iterating over name and language\n*/"]],
                        ["C++", "test.cpp",
                         {"argc": 3, "argv": 3, "mini": 3, "maxi": 3, "std": 2, "atof": 2,
                          "acosd": 2, "main": 1, "zero": 1, "valmax": 1, "exhaust": 1, "test": 1,
                          "pack": 1, "raw": 1},
                         ["// Initialize mini with zero", "/* Initialize maxi with maxval */"]],
                        ["Ruby", "test.rb",
                         {"out": 4, "tmp": 4, "string": 4, "get": 2, "block": 2, "new": 2,
                          "stdout": 1, "stderr": 1},
                         ["# Declaration of get_stdout function", "=begin\nDeal with stderr\n=end"]],
                        ["TypeScript", "test.ts",
                         {"name": 10, "person": 5, "first": 4, "last": 4, "student": 2, "full": 2,
                          "middle": 2, "initi": 2, "greeter": 2, "user": 2, "constructor": 1,
                          "document": 1, "body": 1, "text": 1, "content": 1},
                         ["/** Initialize user instance of Student class */"]],
                        ["TSX", "test.ts",
                         {"name": 10, "person": 5, "first": 4, "last": 4, "student": 2, "full": 2,
                          "middle": 2, "initi": 2, "greeter": 2, "user": 2, "constructor": 1,
                          "document": 1, "body": 1, "text": 1, "content": 1},
                         ["/** Initialize user instance of Student class */"]],
                        ["PHP", "test.php",
                         {"set": 4, "key": 4, "content": 3, "file": 2, "quick": 2, "hash": 2,
                          "int": 2, "get": 1, "dirnam": 1, "load": 1, "from": 1, "string": 1,
                          "not": 1, "donot": 1, "use": 1, "zend": 1, "alloc": 1, "range": 1,
                          "printf": 1, "exists": 1},
                         ["// read file to contents variable",
                          "/*Print information using foreach loop.\n"
                          "On each iteration check set for a key*/"]],
                        ["C#", "test.cs",
                         {"async": 10, "complet": 8, "task": 5, "mytask": 5, "args": 5,
                          "worker": 4, "deleg": 3, "oper": 3, "event": 3, "result": 2,
                          "callback": 1, "iasync": 1, "state": 1, "end": 1, "invoke": 1, "sync": 1,
                          "run": 1, "isrun": 1, "post": 1},
                         ["// get the original worker delegate and the AsyncOperation instance",
                          "// finish the asynchronous operation", "// clear the running task flag",
                          "// raise the completed event"]],
                        ["C", "test.c",
                         {"octalnum": 8, "decimalnum": 4, "temp": 3, "octal": 2, "decim": 2,
                          "todecim": 2, "printf": 2, "pow": 1, "main": 1, "scanf": 1},
                         ['/* This function converts the octal number "octalnum" to the\n'
                          ' * decimal number and returns it.\n */']],
                        ["Scala", "test.scala",
                         {"value": 15, "susp": 9, "println": 8, "lazy": 5, "some": 5, "maybe": 4,
                          "delay": 3, "none": 3, "int": 3, "lib": 2, "impl": 2, "option": 2,
                          "string": 2, "exampl": 1, "force": 1, "function": 1, "apply": 1,
                          "tostr": 1, "evalu": 1, "main": 1, "args": 1, "array": 1},
                         ["// Contributed by John Williams\n", "/*", "*",
                          " Delay the evaluation of an expression until it is needed. ",
                          "*/", "/*", "*", " Get the value of a delayed expression. ", "*/",
                          "/*", "*", " \n   ", "*", " Data type of suspended computations."
                                                    " (The name froms from ML.) \n   ", "*/",
                          "/*", "*", " \n   ", "*", " Implementation of suspended computations,"
                                                    " separated from the \n   ", "*",
                          " abstract class so that the type parameter can be invariant. \n   ",
                          "*/", "// show that s is unevaluated\n", "// evaluate s\n",
                          "// show that the value is saved\n", "// implicit call to force()\n",
                          "// the type is covariant\n"]],
                        ["Shell", "test.sh",
                         {"echo": 10, "file": 8, "rip": 5, "exit": 4, "peaks": 3, "reads": 2,
                          "washu": 2, "root": 2, "tmpdir": 2, "which": 1, "cat": 1, "source": 1,
                          "type": 1, "job": 1, "tmp": 1, "dir": 1, "mkdir": 1},
                         ["# If we already have rip file, do not recalculate"]],
                        ["Rust", "test.rs",
                         {"none": 6, "option": 4, "float": 3, "try": 2, "divis": 2, "println": 2,
                          "unwrap": 2, "main": 1, "equival": 1, "some": 1},
                         ["// Binding `None` to a variable needs to be type annotated",
                          "// Unwrapping a `Some` variant will extract the value wrapped.",
                          "// Unwrapping a `None` variant will `panic!`"]],
                        ["Swift", "test.swift",
                         {"interfac": 11, "address": 8, "target": 5, "first": 2, "nio": 1,
                          "network": 1, "command": 1, "line": 1, "argument": 1, "drop": 1,
                          "socket": 1, "ipaddress": 1, "port": 1, "system": 1, "enumer": 1,
                          "fatal": 1, "error": 1},
                         ["//", "/", " ", "D", "e", "f", "i", "n", "i", "n", "g", " ",
                          "v", "a", "r", "i", "a", "b", "l", "e", " ", "f", "o", "r",
                          " ", "c", "h", "e", "c", "k", "i", "n", "g", " ",
                          "N", "e", "t", "w", "o", "r", "k", "/*", "*", "\n",
                          " ", "C", "h", "e", "c", "k", " ",
                          "i", "n", "t", "e", "r", "f", "a", "c", "e", "\n",
                          " ", "i", "f", " ", "i", "n", "t", "e", "r", "f", "a", "c", "e",
                          " ", "i", "s", " ", "n", "o", "t", " ",
                          "a", "v", "a", "i", "l", "a", "b", "l", "e", " ", "-",
                          " ", "e", "r", "r", "o", "r", " ",
                          "o", "c", "c", "u", "r", "s", "\n", "*/"]],
                        ["Kotlin", "test.kt",
                         {"args": 11, "main": 5, "array": 5, "slice": 4, "until": 4, "size": 4,
                          "project": 2, "extractor": 2, "cli": 1, "string": 1, "empty": 1,
                          "isempti": 1, "println": 1, "trim": 1, "indent": 1, "preprocessor": 1,
                          "parser": 1, "path": 1, "context": 1, "code": 1, "vec": 1, "except": 1},
                         ["// corner-case situation d\n",
                          "/* Process things from args\n"
                          "           If corner-case didn't happend\n         */"]],
                        ["Haskell", "test.hs",
                         {"pat": 10, "match": 6, "lookup": 2, "func": 2, "vars": 2, "just": 2,
                          "noth": 2, "length": 2, "string": 1, "reader": 1, "defs": 1, "ask": 1,
                          "pure": 1, "maybe": 1, "var": 1, "insert": 1, "bpat": 1, "cpat": 1,
                          "when": 1, "fail": 1, "fold": 1, "zip": 1},
                         ["-- functions declaration",
                          "{-", "|Try to match things together|", "-}"]]]

    def test_parser(self):
        for data in TestParser.test_parser_data:
            with self.subTest():
                file = os.path.abspath(os.path.join(tests_dir, "test_files", data[1]))
                tokens = Counter(get_identifiers_sequence_from_file(
                    file, data[0], identifiers_verbose=False, subtokenize=True))
                comments = get_comments_from_file(file)
                self.assertEqual(tokens, Counter(data[2]))
                self.assertEqual(comments, data[3])


if __name__ == "__main__":
    unittest.main()