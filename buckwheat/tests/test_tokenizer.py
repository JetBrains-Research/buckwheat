"""
Parsing-related tests.
"""
from collections import Counter
import os
from typing import List, Dict, Callable, ClassVar
import unittest

from buckwheat.tokenizer import get_identifiers_sequence_from_file, get_comments_from_file

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestSample:
    lang: str
    file: str
    identifiers: Dict[str, int]
    comments: List[str] = None
    exceptions: Dict[Callable, Exception] = None


class TestParser(unittest.TestCase):
    js_test = TestSample()
    js_test.lang = "JavaScript"
    js_test.file = "test.js"
    js_test.identifiers = {"height": 4, "width": 4, "rectangl": 2,
                           "calc": 2, "area": 2, "prototyp": 1}
    js_test.comments = ["// Rectangle instance declaration",
                        "/* Calculating area of Rectangle.\nCreate"
                        " function clacArea.\nArea calculations"
                        " = height * width.\n */"]

    py_test = TestSample()
    py_test.lang = "Python"
    py_test.file = "test.py"
    py_test.identifiers = {"tree": 6, "sitter": 6, "dir": 4, "get": 3, "path": 3,
                           "parser": 2, "str": 2, "bin": 2, "loc": 2, "languag": 1,
                           "abspath": 1, "dirnam": 1, "file": 1, "join": 1}
    py_test.comments = ['# No parsers need to be declared here',
                        '"""\nFunctions definitions below\n"""',
                        '"""\n    Get tree-sitter directory.\n    :return: absolute path.\n'
                        '    """',
                        '"""\n    Get build tree-sitter `.so` location.\n'
                        '    :return: absolute path.\n    """']

    java_test = TestSample()
    java_test.lang = "Java"
    java_test.file = "test.java"
    java_test.identifiers = {"stats": 7, "sub": 7, "dir": 4, "writer": 3, "file": 3,
                             "dirs": 3, "now": 3, "statist": 3, "all": 3, "collect": 2,
                             "write": 2, "list": 2, "featur": 2, "total": 2, "holder": 2,
                             "and": 1, "print": 1, "string": 1, "except": 1, "ioexcept": 1,
                             "length": 1, "directori": 1, "isdirectori": 1, "system": 1,
                             "out": 1, "println": 1, "get": 1, "name": 1, "collector": 1,
                             "from": 1, "project": 1, "path": 1, "topath": 1, "size": 1,
                             "min": 1, "files": 1, "test": 1, "totest": 1}
    java_test.comments = ["/* Function that collects and writes statistics.subDirs\n"
                          "    Uses StatisticsHolder class for statistics information.\n    */",
                          "// total number of subDirs"]


    go_test = TestSample()
    go_test.lang = "Go"
    go_test.file = "test.go"
    go_test.identifiers = {"buff": 4, "file": 4, "fmt": 3, "fprintln": 3, "err": 3,
                           "out": 2, "string": 2, "name": 2, "languag": 2, "version": 1,
                           "commit": 1, "print": 1, "break": 1, "down": 1, "buffer": 1,
                           "data": 1, "read": 1, "limit": 1}
    go_test.comments = ["// Declaration of variables",
                        "/* Breaking Down function\n   iterating over name and language\n*/"]

    cpp_test = TestSample()
    cpp_test.lang = "C++"
    cpp_test.file = "test.cpp"
    cpp_test.identifiers = {"argc": 3, "argv": 3, "mini": 3, "maxi": 3, "std": 2,
                            "atof": 2, "acosd": 2, "main": 1, "zero": 1, "valmax": 1,
                            "exhaust": 1, "test": 1, "pack": 1, "raw": 1}
    cpp_test.comments = ["// Initialize mini with zero", "/* Initialize maxi with maxval */"]

    ruby_test = TestSample()
    ruby_test.lang = "Ruby"
    ruby_test.file = "test.rb"
    ruby_test.identifiers = {"out": 4, "tmp": 4, "string": 4, "get": 2, "block": 2,
                             "new": 2, "stdout": 1, "stderr": 1}
    ruby_test.comments = ["# Declaration of get_stdout function",
                          "=begin\nDeal with stderr\n=end"]

    ts_test = TestSample()
    ts_test.lang = "TypeScript"
    ts_test.file = "test.ts"
    ts_test.identifiers = {"name": 10, "person": 5, "first": 4, "last": 4, "student": 2,
                           "full": 2, "middle": 2, "initi": 2, "greeter": 2, "user": 2,
                           "constructor": 1, "document": 1, "body": 1, "text": 1, "content": 1}
    ts_test.comments = ['// Greeting function',
                        '/** Initialize user instance of Student class */']

    tsx_test = TestSample()
    tsx_test.lang = "TypeScript"
    tsx_test.file = "test.ts"
    tsx_test.identifiers = {"name": 10, "person": 5, "first": 4, "last": 4, "student": 2, "full": 2,
                            "middle": 2, "initi": 2, "greeter": 2, "user": 2, "constructor": 1,
                            "document": 1, "body": 1, "text": 1, "content": 1}
    tsx_test.comments = ['// Greeting function',
                         '/** Initialize user instance of Student class */']

    php_test = TestSample()
    php_test.lang = "PHP"
    php_test.file = "test.php"
    php_test.identifiers = {"set": 4, "key": 4, "content": 3, "file": 2, "quick": 2, "hash": 2,
                            "int": 2, "get": 1, "dirnam": 1, "load": 1, "from": 1, "string": 1,
                            "not": 1, "donot": 1, "use": 1, "zend": 1, "alloc": 1, "range": 1,
                            "printf": 1, "exists": 1}
    php_test.comments = ["// read file to contents variable",
                         "/*Print information using foreach loop.\n"
                         "On each iteration check set for a key*/"]

    cs_test = TestSample()
    cs_test.lang = "C#"
    cs_test.file = "test.cs"
    cs_test.identifiers = {"async": 10, "complet": 8, "task": 5, "mytask": 5, "args": 5,
                           "worker": 4, "deleg": 3, "oper": 3, "event": 3, "result": 2,
                           "callback": 1, "iasync": 1, "state": 1, "end": 1, "invoke": 1, "sync": 1,
                           "run": 1, "isrun": 1, "post": 1}
    cs_test.comments = ['// get the original worker delegate and the AsyncOperation instance',
                        '// finish the asynchronous operation', '// clear the running task flag',
                        '/* raise the\n  completed event*/']

    c_test = TestSample()
    c_test.lang = "C"
    c_test.file = "test.c"
    c_test.identifiers = {"octalnum": 8, "decimalnum": 4, "temp": 3, "octal": 2, "decim": 2,
                          "todecim": 2, "printf": 2, "pow": 1, "main": 1, "scanf": 1}
    c_test.comments = ['/* This function converts the octal number "octalnum" to the\n'
                       ' * decimal number and returns it.\n */',
                       '// main function - the starting point']

    scala_test = TestSample()
    scala_test.lang = "Scala"
    scala_test.file = "test.scala"
    scala_test.identifiers = {"value": 15, "susp": 9, "println": 8, "lazy": 5, "some": 5,
                              "maybe": 4, "delay": 3, "none": 3, "int": 3, "lib": 2,
                              "impl": 2, "option": 2, "string": 2, "exampl": 1, "force": 1,
                              "function": 1, "apply": 1, "tostr": 1, "evalu": 1,
                              "main": 1, "args": 1, "array": 1}
    scala_test.comments = ["// Contributed by John Williams\n", "/*", "*",
                           " Delay the evaluation of an expression until it is needed. ",
                           "*/", "/*", "*", " Get the value of a delayed expression. ", "*/",
                           "/*", "*", " \n   ", "*", " Data type of suspended computations."
                                                     " (The name froms from ML.) \n   ", "*/",
                           "/*", "*", " \n   ", "*", " Implementation of suspended computations,"
                                                     " separated from the \n   ", "*",
                           " abstract class so that the type parameter can be invariant. \n   ",
                           "*/", "// show that s is unevaluated\n", "// evaluate s\n",
                           "// show that the value is saved\n", "// implicit call to force()\n",
                           "// the type is covariant\n"]

    shell_test = TestSample()
    shell_test.lang = "Shell"
    shell_test.file = "test.sh"
    shell_test.identifiers = {"echo": 10, "file": 8, "rip": 5, "exit": 4, "peaks": 3,
                              "reads": 2, "washu": 2, "root": 2, "tmpdir": 2, "which": 1, "cat": 1,
                              "source": 1, "type": 1, "job": 1, "tmp": 1, "dir": 1, "mkdir": 1}
    shell_test.comments = ["# If we already have rip file, do not recalculate"]

    rust_test = TestSample()
    rust_test.lang = "Rust"
    rust_test.file = "test.rs"
    rust_test.identifiers = {"none": 6, "option": 4, "float": 3, "try": 2, "divis": 2,
                             "println": 2, "unwrap": 2, "main": 1, "equival": 1, "some": 1}
    rust_test.comments = ['// Binding `None` to a variable needs to be type annotated',
                          '// Unwrapping a `Some` variant will extract the value wrapped.',
                          '/*\n     Unwrapping a `None`\n     variant will `panic!`\n     */']

    swift_test = TestSample()
    swift_test.lang = "Swift"
    swift_test.file = "test.swift"
    swift_test.identifiers = {"interfac": 11, "address": 8, "target": 5, "first": 2, "nio": 1,
                              "network": 1, "command": 1, "line": 1, "argument": 1, "drop": 1,
                              "socket": 1, "ipaddress": 1, "port": 1, "system": 1,
                              "enumer": 1, "fatal": 1, "error": 1}
    swift_test.exceptions = {get_comments_from_file: ValueError}

    kotlin_test = TestSample()
    kotlin_test.lang = "Kotlin"
    kotlin_test.file = "test.kt"
    kotlin_test.identifiers = {"args": 11, "main": 5, "array": 5, "slice": 4, "until": 4,
                               "size": 4, "project": 2, "extractor": 2, "cli": 1, "string": 1,
                               "empty": 1, "isempti": 1, "println": 1, "trim": 1, "indent": 1,
                               "preprocessor": 1, "parser": 1, "path": 1, "context": 1,
                               "code": 1, "vec": 1, "except": 1}
    kotlin_test.comments = ["// corner-case situation d\n",
                            "/* Process things from args\n"
                            "           If corner-case didn't happend\n         */"]

    haskell_test = TestSample()
    haskell_test.lang = "Haskell"
    haskell_test.file = "test.hs"
    haskell_test.identifiers = {"pat": 10, "match": 6, "lookup": 2, "func": 2, "vars": 2,
                                "just": 2, "noth": 2, "length": 2, "string": 1, "reader": 1,
                                "defs": 1, "ask": 1, "pure": 1, "maybe": 1, "var": 1, "insert": 1,
                                "bpat": 1, "cpat": 1, "when": 1, "fail": 1, "fold": 1, "zip": 1}
    haskell_test.comments = ["-- functions declaration",
                             "{-", "|Try to match things together|", "-}"]

    test_parser_data = [js_test, py_test, java_test, go_test, cpp_test, ruby_test,
                        ts_test, tsx_test, php_test, cs_test, c_test, scala_test,
                        shell_test, rust_test, swift_test, kotlin_test, haskell_test]

    def test_parser(self):
        for data in TestParser.test_parser_data:
            with self.subTest():
                file = os.path.abspath(os.path.join(tests_dir, "test_files", data.file))
                tokens = Counter(get_identifiers_sequence_from_file(
                    file, data.lang, identifiers_verbose=False, subtokenize=True))
                self.assertEqual(tokens, Counter(data.identifiers))

                if data.comments:
                    comments = get_comments_from_file(file)
                    self.assertEqual(comments, data.comments)

                if data.exceptions:
                    for function in data.exceptions:
                        with self.assertRaises(data.exceptions[function]):
                            function(file)


if __name__ == "__main__":
    unittest.main()
