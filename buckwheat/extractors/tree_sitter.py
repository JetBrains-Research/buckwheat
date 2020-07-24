from dataclasses import dataclass
from typing import Generator, Set

import tree_sitter

from buckwheat.extractors.base import BaseEntityExtractor
from buckwheat.extractors.entities import BaseEntity, TraversableEntity
from buckwheat.parsing.utils import get_parser

PARSERS = {"JavaScript": "javascript",
           "Python": "python",
           "Java": "java",
           "Go": "go",
           "C++": "cpp",
           "Ruby": "ruby",
           "TypeScript": "typescript",
           "TSX": "tsx",
           "PHP": "php",
           "C#": "c_sharp",
           "C": "c",
           "Shell": "bash",
           "Rust": "rust"}


# Tree-sitter nodes corresponding to identifiers in a given language.
IDENTIFIERS = {"JavaScript": {"identifier", "property_identifier",
                              "shorthand_property_identifier"},
               "Python": {"identifier"},
               "Java": {"identifier", "type_identifier"},
               "Go": {"identifier", "field_identifier", "type_identifier"},
               "C++": {"identifier", "namespace_identifier", "field_identifier",
                       "type_identifier"},
               "Ruby": {"identifier", "constant", "symbol"},
               "TypeScript": {"identifier", "property_identifier",
                              "shorthand_property_identifier", "type_identifier"},
               "TSX": {"identifier", "property_identifier",
                       "shorthand_property_identifier", "type_identifier"},
               "PHP": {"name"},
               "C#": {"identifier"},
               "C": {"identifier", "field_identifier", "type_identifier"},
               "Shell": {"variable_name", "command_name"},
               "Rust": {"identifier", "field_identifier", "type_identifier"}}


# Tree-sitter nodes corresponding to classes in a given language.
CLASSES = {"JavaScript": {"class_declaration"},
           "Python": {"class_definition"},
           "Java": {"class_declaration"},
           "C++": {"class_specifier"},
           "Ruby": {"class"},
           "TypeScript": {"class_declaration"},
           "TSX": {"class_declaration"},
           "PHP": {"class_declaration"},
           "C#": {"class_declaration"}}


# Tree-sitter nodes corresponding to functions in a given language.
FUNCTIONS = {"JavaScript": {"function", "function_declaration", "method_definition"},
             "Python": {"function_definition"},
             "Java": {"constructor_declaration", "method_declaration",
                      "interface_declaration"},
             "Go": {"function_declaration", "method_declaration"},
             "C++": {"function_definition"},
             "Ruby": {"method", "singleton_method"},
             "TypeScript": {"function", "function_declaration", "method_definition"},
             "TSX": {"function", "function_declaration", "method_definition"},
             "PHP": {"function_definition", "method_declaration"},
             "C#": {"method_declaration", "indexer_declaration", "property_declaration",
                    "constructor_declaration"},
             "C": {"function_definition"},
             "Shell": {"function_definition"},
             "Rust": {"function_item"}}


@dataclass
class TreeSitterExtractor(BaseEntityExtractor):
    """Entities extractor with tree-sitter backend"""
    types: Set[str]

    def __post_init__(self):
        parser_name = PARSERS[self.programming_language.value]
        self.parser = get_parser(parser_name)

    def traverse_tree(self, code: str) -> Generator[tree_sitter.Node, None, None]:
        """
        Traverse tree with TreeCursor in DFS-order and yield nodes of given types

        :param code: source code string
        :return: generator of tree_sitter.Node instances with given types
        """
        tree: tree_sitter.Tree = self.parser.parse(code.encode())
        cursor: tree_sitter.TreeCursor = tree.walk()

        has_next_child = True
        has_next_sibling = True
        has_parent_node = True

        while has_next_child or has_next_sibling:
            if cursor.node.type in self.types:
                yield cursor.node

            # Traverse down
            has_next_child = cursor.goto_first_child()

            # If leaf node is met try traverse right or find parent node where we can traverse right
            if not has_next_child:
                has_next_sibling = cursor.goto_next_sibling()

                while not has_next_sibling and has_parent_node:
                    has_parent_node = cursor.goto_parent()
                    has_next_sibling = cursor.goto_next_sibling()

    def parse_entities(self, code: str) -> Generator[BaseEntity, None, None]:
        """
        Parse entities from code with tree_sitter extractor.

        :param code: source code string
        :return: entities with self.types
        """
        code_bytes = code.encode()
        for node in self.traverse_tree(code):
            identifier = code_bytes[node.start_byte:node.end_byte].decode()
            yield BaseEntity(identifier, node.start_byte, node.type)

    def parse_traversable_entities(self, code: str):
        """
        Parse traversable entities from code with pygments extractor. Differs from parse_entities
        with presence of node in each entity, which allows to make something with AST around the node.

        :param code: source code string
        :return: entities with self.types
        """
        code_bytes = code.encode()
        for node in self.traverse_tree(code):
            identifier = code_bytes[node.start_byte:node.end_byte].decode()
            yield TraversableEntity(identifier, node.start_byte, node.type, node)
