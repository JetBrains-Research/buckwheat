"""
Tree-sitter related functionality.
"""
import os
import urllib.request

from tree_sitter import Language, Parser

PARSERS = {}


def get_tree_sitter_dir() -> str:
    """
    Get tree-sitter directory.
    :return: absolute path.
    """
    return os.path.abspath(os.path.dirname(__file__))


def get_tree_sitter_so() -> str:
    """
    Get build tree-sitter `.so` location.
    :return: absolute path.
    """
    tree_sitter_dir = get_tree_sitter_dir()
    bin_loc = os.path.join(tree_sitter_dir, "build", "langs.so")
    return bin_loc


def main() -> None:
    """
    Initialize tree-sitter library.
    :return: None.
    """
    url = "https://github.com/areyde/buckwheat/releases/download/v1.1.1/langs.so"
    filename = "langs.so"
    if not os.path.exists(get_tree_sitter_so()):
        urllib.request.urlretrieve(url,
                                   os.path.abspath(os.path.join(get_tree_sitter_dir(), "build",
                                                                filename)))
    print("Parser successfully initialized.")


def get_parser(lang: str) -> Parser:
    """
    Initialize parser for a specific language.
    :param lang: language to use.
    :return: parser.
    """
    global PARSERS
    if lang not in PARSERS:
        parser = Parser()
        parser.set_language(Language(get_tree_sitter_so(), lang))
        PARSERS[lang] = parser
    else:
        parser = PARSERS[lang]
    return parser
