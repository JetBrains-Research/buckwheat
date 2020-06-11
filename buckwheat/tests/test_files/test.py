import os

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
    bin_loc = os.path.join(tree_sitter_dir, "build/langs.so")
    return bin_loc
