"""
Parsing-related functionality.
"""

from typing import List, Tuple

import tree_sitter
from .parsers.utils import get_parser
from collections import Counter
from operator import itemgetter


def read_file(file: str) -> bytes:
    """
    Read the contents of the file.
    :param: file: address of the file.
    :return: bytes with the contents of the file.
    """
    with open(file, 'r') as fin:
        code = bytes(fin.read(), 'utf-8')
    return code


def get_positional_bytes(node: tree_sitter.Node) -> Tuple[int, int]:
    """
    Extract start and end byte.
    :param node: node on the AST.
    :return: (start byte, end byte)
    """
    start = node.start_byte
    end = node.end_byte
    return start, end


def get_identifiers(file: str, lang: str) -> List:
    """
    Gather a sorted list of identifiers in the file and their count.
    :param file: address of the file.
    :param lang: the language of file.
    :return: a list of identifiers.
    """
    code = read_file(file)
    tree = get_parser(lang).parse(code)
    root = tree.root_node
    identifiers = []
    node_types = {'c' : ['identifier', 'type_identifier'],
                  'c-sharp' : ['identifier', 'type_identifier'],
                  'cpp' : ['identifier', 'type_identifier'],
                  'java' : ['identifier', 'type_identifier'],
                  'python' : ['identifier', 'type_identifier']}

    def traverse_tree(node: tree_sitter.Node) -> None:
        """
        Run down the AST from a given node and gather identifiers from its childern.
        :param node: starting node.
        :return: None.
        """
        for child in node.children:
            if child.type in node_types[lang]:
                start, end = get_positional_bytes(child)
                identifiers.append(code[start:end].decode('utf-8'))
            if len(child.children) != 0:
                traverse_tree(child)

    traverse_tree(root)
    sorted_identifiers = sorted(Counter(identifiers).items(), key = itemgetter(1), reverse=True)

    return sorted_identifiers