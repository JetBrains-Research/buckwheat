from dataclasses import dataclass
from typing import Generator

from buckwheat.extractors.entities import BaseEntity
from buckwheat.utils import ProgrammingLanguages


@dataclass
class BaseEntityExtractor:
    """Base interface for entity extractors, defines function signature that yields extracted entities from code"""
    programming_language: ProgrammingLanguages

    def parse_entities(self, code: str) -> Generator[BaseEntity, None, None]:
        """
        Parse entities from code

        :param code: string with code file to parse
        :return: generator of BaseEntities
        """
        raise NotImplementedError
