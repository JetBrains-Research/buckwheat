from dask import bag

from buckwheat.extractors.tree_sitter import MERGED_CLASSES, MERGED_FUNCTIONS
from buckwheat.pipeline.premade import extract_identifiers_from_projects, extract_identifiers_for_classes, \
    extract_identifiers_for_functions, extract_identifiers_from_files
from tests.base import get_directory_instance, get_classified_test_files, get_test_files_list


# TODO: More specific test cases
def test_extract_identifiers_from_projects():
    projects_bag = bag.from_sequence([get_directory_instance()])
    project_identifiers = list(extract_identifiers_from_projects(projects_bag))
    assert len(project_identifiers) == len(list(projects_bag))
    assert all([len(identifiers) > 0 for _, identifiers in project_identifiers])


def test_extract_identifiers_from_files():
    files_bag = bag.from_sequence(get_classified_test_files())
    file_identifiers_bag = list(extract_identifiers_from_files(files_bag))
    extracted_files = set(file.path for file, _ in file_identifiers_bag)
    actual_files = set(get_test_files_list())
    assert extracted_files == actual_files
    assert all([len(identifiers) > 0 for _, identifiers in file_identifiers_bag])


def test_extract_identifiers_from_functions():
    files_bag = bag.from_sequence(get_classified_test_files())
    file_functions = extract_identifiers_for_functions(files_bag)
    assert any([len(identifiers) > 0 for _, identifiers in file_functions])
    assert all([identifier.type in MERGED_FUNCTIONS for _, identifiers in file_functions for identifier in identifiers])
    assert all([len(identifier.children) > 0 for _, identifiers in file_functions for identifier in identifiers])


def test_extract_identifiers_from_classes():
    files_bag = bag.from_sequence(get_classified_test_files())
    file_classes = extract_identifiers_for_classes(files_bag)
    assert any([len(identifiers) > 0 for _, identifiers in file_classes])
    assert all([identifier.type in MERGED_CLASSES for _, identifiers in file_classes for identifier in identifiers])
    assert all([len(identifier.children) > 0 for _, identifiers in file_classes for identifier in identifiers])
