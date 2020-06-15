"""
Dataclasses for working with data
"""
from collections import Counter
import dataclasses
from typing import List, Optional, Union


@dataclasses.dataclass
class IdentifierData:
    """
    Data class to store individual identifiers and their positional coordinates.
    """
    identifier: str
    start_byte: int
    start_line: int
    start_column: int


@dataclasses.dataclass
class ObjectData:
    """
    Data class to store objects (classes and functions) and their parameters: positional
    coordinates, language and identifiers.
    """
    object_type: str
    content: str
    lang: str
    identifiers: Union[List[IdentifierData], List[str]]
    start_byte: int
    start_line: int
    start_column: int
    end_byte: int
    end_line: int
    end_column: int


@dataclasses.dataclass
class FileData:
    """
    Dataclass to store files and their content.
    """
    path: str
    lang: str
    objects: List[ObjectData]
    identifiers: Union[List[IdentifierData], List[str]]
