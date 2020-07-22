from typing import List, Callable, Set, Dict

from buckwheat.extractors import BaseEntityExtractor
from buckwheat.extractors.entities import BaseEntity, TreeEntity
from buckwheat.extractors.pygments import IDENTIFIERS as PYGMENTS_IDENTIFIERS, PygmentsExtractor
from buckwheat.extractors.tree_sitter import IDENTIFIERS as TREE_SITTER_IDENTIFIERS, CLASSES, FUNCTIONS, \
    TreeSitterExtractor
from buckwheat.pipeline.input import LanguageClassifiedFile
from buckwheat.utils import ProgrammingLanguages

ExtractorFunction = Callable[[LanguageClassifiedFile], List[BaseEntity]]
TreeExtractorFunction = Callable[[LanguageClassifiedFile], List[TreeEntity]]


def compose_extractors(extractors: List[BaseEntityExtractor]) -> ExtractorFunction:
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
                if extractor.programming_language == file.language
                for entity in extractor.parse_entities(code)
            ]

    return entity_extractor


def _build_identifiers_extractor() -> ExtractorFunction:
    """
    Build identifiers extraction function from all available extractors for languages

    :return: identifier extraction function
    """
    extractors: List[BaseEntityExtractor] = []

    for language in ProgrammingLanguages:
        if language.value in TREE_SITTER_IDENTIFIERS:
            types = TREE_SITTER_IDENTIFIERS[language.value]
            extractors.append(TreeSitterExtractor(language, types))
        elif language.value in PYGMENTS_IDENTIFIERS:
            types = PYGMENTS_IDENTIFIERS[language.value]
            extractors.append(PygmentsExtractor(language, types))

    return compose_extractors(extractors)


# Pre-made identifiers extractor with pygments and tree-sitter
identifiers_extractor = _build_identifiers_extractor()


def compose_structural_extractors(extractors: List[TreeSitterExtractor]) -> TreeExtractorFunction:
    """
    Compose extractors for TreeEntities from code.

    :param extractors: list of entity extractors to use on files
    :return: entity extractor that converts file to list of entities
    """

    def entity_extractor(file: LanguageClassifiedFile) -> List[TreeEntity]:
        """
        Extract entities from file

        :param file: file with classified language and code
        :return: list of entities
        """
        with open(file.path, "r", encoding="utf-8") as f:
            code = f.read()
            return [
                entity
                for extractor in extractor
                if extractor.programming_language == file.language
                for entity in extractor.parse_entities_with_children(code)
            ]

    return entity_extractor


def compose_extractors_with_childrens_of_types(extractors: List[TreeSitterExtractor], children_types: Set[str]) -> TreeExtractorFunction:
    """
    Compose extractors for traversable entities from code.

    :param extractors: list of entity extractors to use on files
    :return: entity extractor that converts file to list of entities
    """

    def entity_extractor(file: LanguageClassifiedFile) -> List[TreeEntity]:
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
                if extractor.programming_language == file.language
                for entity in extractor.parse_entities_with_children_of_types(code, children_types)
            ]

    return entity_extractor


def build_extractors_from_type_spec(type_spec: Dict[str, Set[str]]) -> List[TreeSitterExtractor]:
    """
    Return list of extractors constructed from type spec.

    :param type_spec: dictionary with programming languages as keys and tree_sitter types as values
    :return: list of extractors used for code extraction
    """
    return [TreeSitterExtractor(ProgrammingLanguages(language), types) for language, types in type_spec.items()]


# Pre-made lists of extractors for types specified in extractors.tree_sitter
TS_IDENTIFIERS_EXTRACTORS = build_extractors_from_type_spec(TREE_SITTER_IDENTIFIERS)

TS_FUNCTION_EXTRACTORS = build_extractors_from_type_spec(FUNCTIONS)

TS_CLASSES_EXTRACTORS = build_extractors_from_type_spec(CLASSES)
