from typing import List, Set

import pytest

from buckwheat.extractors import BaseEntityExtractor
from buckwheat.extractors.tree_sitter import TreeSitterExtractor, FUNCTIONS, IDENTIFIERS
from buckwheat.pipeline.extraction import compose_extractors
from buckwheat.utils import ProgrammingLanguages
from tests.base import get_classified_test_files


@pytest.mark.parametrize(
    "types_to_extract,extracted_languages,extractors",
    [
        (
                IDENTIFIERS[ProgrammingLanguages.PYTHON.value] | FUNCTIONS[ProgrammingLanguages.PYTHON.value],
                {ProgrammingLanguages.PYTHON},
                [TreeSitterExtractor(ProgrammingLanguages.PYTHON, FUNCTIONS[ProgrammingLanguages.PYTHON.value]),
                 TreeSitterExtractor(ProgrammingLanguages.PYTHON, IDENTIFIERS[ProgrammingLanguages.PYTHON.value])],
        ),
        (
                IDENTIFIERS[ProgrammingLanguages.PYTHON.value] | IDENTIFIERS[ProgrammingLanguages.C.value],
                {ProgrammingLanguages.PYTHON, ProgrammingLanguages.C},
                [TreeSitterExtractor(ProgrammingLanguages.C, IDENTIFIERS[ProgrammingLanguages.C.value]),
                 TreeSitterExtractor(ProgrammingLanguages.PYTHON, IDENTIFIERS[ProgrammingLanguages.PYTHON.value])],
        ),
    ]
)
def test_compose_extractors(types_to_extract: Set[str], extracted_languages: Set[ProgrammingLanguages],
                            extractors: List[BaseEntityExtractor]):
    test_files = get_classified_test_files()
    composite_extractor = compose_extractors(extractors)

    for file in test_files:
        entities = composite_extractor(file)
        assert (file.language in extracted_languages) == bool(entities)
        assert all([entity.type in types_to_extract for entity in entities])
