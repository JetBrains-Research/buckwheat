from dataclasses import dataclass
from typing import Generator

from buckwheat.extractors.entities import BaseEntity
from buckwheat.utils import ProgrammingLanguages


@dataclass
class BaseEntityExtractor:
    programming_language: ProgrammingLanguages

    def parse_entities(self, code: str) -> Generator[BaseEntity, None, None]:
        raise NotImplementedError
