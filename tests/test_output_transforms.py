from typing import Callable

import pytest

from buckwheat.extractors.entities import BaseEntity
from buckwheat.pipeline.input import LanguageClassifiedFile
from buckwheat.pipeline.output import build_identifiers_to_vawbit_transformer, entity_to_wabbit, \
    entity_to_wabbit_verbose
from buckwheat.utils import ProgrammingLanguages


def test_verbose_wabbit_representation():
    entity = BaseEntity("test", 0, 1, 2, "test")
    assert entity_to_wabbit_verbose(entity) == f"test:0,1,2"


def test_wabbit_representation():
    entity = BaseEntity("test", 0, 0, 0, "test")
    assert entity_to_wabbit(entity) == entity.body


@pytest.mark.parametrize("verbose,repr_function", [(True, entity_to_wabbit_verbose), (False, entity_to_wabbit)])
def test_multiple_entities_wabbit_representation(verbose: bool, repr_function: Callable[[str], str]):
    file = LanguageClassifiedFile("some_test.py", ProgrammingLanguages.PYTHON)
    entities = [
        BaseEntity("test1", 0, 0, 0, "test"),
        BaseEntity("test2", 1, 2, 3, "test"),
        BaseEntity("test3", 4, 5, 6, "test"),
    ]
    output_transformer = build_identifiers_to_vawbit_transformer(verbose)
    output = output_transformer((file, entities))
    entities = " ".join(repr_function(entity) for entity in entities)
    assert output == f"{file.path} {entities}"
