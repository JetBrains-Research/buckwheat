"""
Dataclasses for working with data
"""
import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class IdentifierData:
    """
    Data class to store individual identifiers and their positional coordinates.
    """
    identifier: str
    start_byte: int
    start_line: int
    start_column: int


@dataclasses.dataclass(frozen=True)
class ObjectData:
    """
    Data class to store objects (classes and functions) and their parameters: positional
    coordinates, language and identifiers.
    """
    content: str
    identifiers: Tuple[IdentifierData]
    start_byte: int
    start_line: int
    start_column: int
    end_byte: int
    end_line: int
    end_column: int
