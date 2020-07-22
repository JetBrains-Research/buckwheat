from dataclasses import dataclass
from typing import List, Set

import tree_sitter

from buckwheat.extractors.tokenizer import TokenParser
from buckwheat.extractors.utils import traverse_down_with_cursor

tokenizer = TokenParser()


@dataclass
class BaseEntity:
    """Entity from source code with start and end position"""
    body: str
    start_byte: int
    start_line: int
    start_column: int
    type: str

    # TODO: Check if identifier
    @property
    def subtokens(self) -> List[str]:
        """Return tokens from entity body"""
        return tokenizer.process_token(self.body)


@dataclass
class TreeEntity(BaseEntity):
    """Entity with tree representation, suitable for extracting graphlets and nested entities"""
    children: List["TreeEntity"]

    @classmethod
    def from_node(cls, node: tree_sitter.Node, code_bytes: bytes) -> "TreeEntity":
        """
        Construct TreeEntity from node and code_bytes. Extract all children for entity as presented in AST.

        :param node: tree_sitter node instance for extraction
        :param code_bytes: code bytes to extract body from
        :return: constructed tree entity
        """
        body = code_bytes[node.start_byte:node.end_byte].decode()
        start_line, start_column = node.start_point
        children = []
        cursor = node.walk()

        has_children = cursor.goto_first_child()
        has_siblings = True

        while has_children and has_siblings:
            children.append(cls.from_node(cursor.node, code_bytes))
            has_siblings = cursor.goto_next_sibling()

        return cls(body, node.start_byte, start_line, start_column, node.type, children)

    @classmethod
    def from_node_with_types(cls, node: tree_sitter.Node, code_bytes: bytes, types: Set[str]) -> "TreeEntity":
        """
        Construct TreeEntity from node and code_bytes. Childrens are extracted to flat list in DFS-order.
        This should be used for extracting entities grouped by parents.

        :param node: tree_sitter node instance
        :param code_bytes: code bytes used to extract entities bodies
        :param types: types of nodes extracted from node descedants
        :return: TreeEntity with flat children list of given type
        """
        cursor = node.walk()
        body = code_bytes[node.start_byte:node.end_byte].decode()
        children = [
            cls(code_bytes[cnode.start_byte:cnode.end_byte].decode(), cnode.start_byte, cnode.start_point[0],
                cnode.start_point[1], cnode.type, [])
            for cnode in traverse_down_with_cursor(cursor, types)
        ]
        return cls(body, node.start_byte, node.start_point[0], node.start_point[1], node.type, children)
