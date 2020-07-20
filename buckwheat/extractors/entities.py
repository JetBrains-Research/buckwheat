from dataclasses import dataclass
from typing import List

import tree_sitter

from buckwheat.extractors.tokenizer import TokenParser

tokenizer = TokenParser()


@dataclass
class BaseEntity:
    """Entity from source code with start and end position"""
    body: str
    start_byte: int
    type: str

    # TODO: Check if identifier
    @property
    def subtokens(self) -> List[str]:
        return tokenizer.process_token(self.body)


@dataclass
class TraversableEntity(BaseEntity):
    """Entity from source code with possibility to traverse from ast node"""
    node: tree_sitter.Node
