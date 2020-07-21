from typing import Tuple, List, Callable

from buckwheat.extractors.entities import BaseEntity
from buckwheat.pipeline.input import LanguageClassifiedFile

FileEntitiesPair = Tuple[LanguageClassifiedFile, List[BaseEntity]]


def entity_to_wabbit(entity: BaseEntity) -> str:
    """Construct wabbit entity representation"""
    return entity.body


def entity_to_wabbit_verbose(entity: BaseEntity) -> str:
    """Construct verbose wabbit entity representations"""
    return f"{entity.body}:{entity.start_byte},{entity.start_line},{entity.start_column}"


def build_identifiers_to_vawbit_transformer(verbose: bool = True) -> Callable[[FileEntitiesPair], str]:
    """
    Build function to transform parsed nodes into vowpal wabbit format
    More information about format can be found at https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format

    :param verbose: If True verbose format representation used otherwise plain identifiers are used
    :return: transformer function
    """

    def identifiers_to_vawbit(file_with_entities: FileEntitiesPair) -> str:
        """
        Construct compact vowpal wabbit string representation from list of parsed nodes, filename of representation is
        the first node file path.

        :param file_with_entities: (file, entities) tuple where file is source_code file and entities are entities
        :return: compact string representation of file
        """
        file, entities = file_with_entities
        identifiers_string = " ".join([entity_to_wabbit(entity) for entity in entities])
        return f"{file.path} {identifiers_string}"

    def verbose_identifiers_to_vawbit(file_with_entities: FileEntitiesPair) -> str:
        """
        Construct verbose vowpal wabbit string representation from list of parsed nodes, filename of representation is
        the first node file path.

        :param file_with_entities: (file, entities) tuple where file is source_code file and entities are entities
        :return: verbose string representation of file
        """
        file, entities = file_with_entities
        identifiers_string = " ".join([entity_to_wabbit_verbose(entity) for entity in entities])
        return f"{file.path} {identifiers_string}"

    if verbose:
        return verbose_identifiers_to_vawbit

    return identifiers_to_vawbit
