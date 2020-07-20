from dataclasses import dataclass
from typing import Generator, Set, Iterable, Any

import pygments
from pygments.lexers.haskell import HaskellLexer
from pygments.lexers.jvm import ScalaLexer, KotlinLexer
from pygments.lexers.objective import SwiftLexer

from buckwheat.extractors.entities import BaseEntity
from buckwheat.extractors.base import BaseEntityExtractor


IDENTIFIERS = {
    "Scala": {pygments.token.Name, pygments.token.Keyword.Type},
    "Swift": {pygments.token.Name},
    "Kotlin": {pygments.token.Name},
    "Haskell": {pygments.token.Name, pygments.token.Keyword.Type},
}


LEXERS = {
    "Scala": ScalaLexer(),
    "Swift": SwiftLexer(),
    "Kotlin": KotlinLexer(),
    "Haskell": HaskellLexer(),
}


@dataclass
class PygmentsExtractor(BaseEntityExtractor):
    """
    Entities extractor that uses pygments for extraction.
    This extractor can only extract non-traversable entities and no structural information about code.
    """
    types: Set[Iterable[Any]]

    def __post_init__(self):
        self.lexer = LEXERS[self.programming_language.value]

    def parse_entities(self, code: str) -> Generator[BaseEntity, None, None]:
        for index, token_type, token in self.lexer.get_tokens_unprocessed(code):
            if any(token_type in token_types for token_types in self.types):
                start_index = index
                yield BaseEntity(token, start_index, token_type)
