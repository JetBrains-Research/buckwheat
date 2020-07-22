from dataclasses import dataclass
from typing import Set, Generator, Dict

import tree_sitter

from buckwheat.extractors.entities import BaseEntity, TreeEntity
from buckwheat.extractors.base import BaseEntityExtractor
from buckwheat.extractors.utils import traverse_down_with_cursor
from buckwheat.parsing.utils import get_parser


# Tree-sitter parsers corresponding to given language
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
             "TypeScript": {"function_declaration", "method_definition"},
             "TSX": {"function_declaration", "method_definition"},
             "PHP": {"function_definition", "method_declaration"},
             "C#": {"method_declaration", "indexer_declaration", "property_declaration",
                    "constructor_declaration"},
             "C": {"function_definition"},
             "Shell": {"function_definition"},
             "Rust": {"function_item"}}


def merge_types_for_all_languages(type_spec: Dict[str, Set[str]]) -> Set[str]:
    """
    Merge types for all languages presented in type_spec into one set.

    :param type_spec: dict with programming languages as keys and set of tree sitter types as keys
    :return: set of tree sitter types
    """
    return set(value for values in type_spec.values() for value in values)


# Merged types for all languages for all specs defined above
MERGED_IDENTIFIERS = merge_types_for_all_languages(IDENTIFIERS)

MERGED_CLASSES = merge_types_for_all_languages(CLASSES)

MERGED_FUNCTIONS = merge_types_for_all_languages(FUNCTIONS)


@dataclass
class TreeSitterExtractor(BaseEntityExtractor):
    """Entities extractor with tree-sitter backend"""
    types: Set[str]

    @property
    def parser(self) -> tree_sitter.Parser:
        """Return parser for currently specified language for extraction"""
        parser_name = PARSERS[self.programming_language.value]
        return get_parser(parser_name)

    def traverse_tree(self, code: str) -> Generator[tree_sitter.Node, None, None]:
        """
        Traverse tree with TreeCursor in DFS-order and yield nodes of given types

        :param code: source code string
        :return: generator of tree_sitter.Node instances with given types
        """
        tree: tree_sitter.Tree = self.parser.parse(code.encode())
        yield from traverse_down_with_cursor(tree.walk(), types=self.types)

    def parse_entities(self, code: str) -> Generator[BaseEntity, None, None]:
        """
        Parse entities from code with tree_sitter extractor.

        :param code: source code string
        :return: entities with self.types
        """
        code_bytes = code.encode()
        for node in self.traverse_tree(code):
            identifier = code_bytes[node.start_byte:node.end_byte].decode()
            yield BaseEntity(identifier, node.start_byte, *node.start_point, node.type)

    def parse_entities_with_children(self, code: str) -> Generator[TreeEntity, None, None]:
        """
        Parse TreeEntity from code with tree_sitter extractor.
        Differs from parse_entity in that structural information is extracted.

        :param code: source code string
        :return: generator of TreeEntities
        """
        code_bytes = code.encode()
        for node in self.traverse_tree(code):
            yield TreeEntity.from_node(node, code_bytes)

    def parse_entities_with_children_of_types(self, code: str, children_types: Set[str]) -> Generator[TreeEntity, None, None]:
        """
        Parse TreeEntity from code with tree_sitter extractor.
        Differs from parse_entities_with_children in that only children with defined children_types are extracted. Also
        structural information aren't preserved and children come flat.

        :param code: source code string
        :param children_types: type of children to extract
        :return: generator of TreeEntities
        """
        code_bytes = code.encode()
        for node in self.traverse_tree(code):
            entity = TreeEntity.from_node_with_types(node, code_bytes, children_types)
            if len(entity.children) > 0:
                yield entity
