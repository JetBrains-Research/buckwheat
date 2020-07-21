from typing import List

from buckwheat.extractors import BaseEntityExtractor
from buckwheat.extractors.entities import BaseEntity
from buckwheat.pipeline.input import LanguageClassifiedFile


def compose_extractors(extractors: List[BaseEntityExtractor]):
    """
    Compose multiple extractors into extractor function to extract entities from multiple-language files or
    extract multiple types of entities

    :param extractors: list of entity extractors to use on files
    :return: entity extractor that converts file to list of entities
    """

    def entity_extractor(file: LanguageClassifiedFile) -> List[BaseEntity]:
        """
        Extract entities from file

        :param file: file with classified language and code
        :return: list of entities
        """
        with open(file.path, "r", encoding="utf-8") as f:
            code = f.read()
            return [
                entity
                for extractor in extractors
                for entity in extractor.parse_entities(code)
                if extractor.programming_language == file.language
            ]

    return entity_extractor
