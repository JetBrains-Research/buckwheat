# TODO: Better naming for this module
from typing import List, Tuple, TypeVar

from dask import bag

from buckwheat.extractors.entities import BaseEntity
from buckwheat.extractors.tree_sitter import MERGED_IDENTIFIERS, TreeSitterExtractor
from buckwheat.pipeline.extraction import identifiers_extractor, compose_extractors_with_childrens_of_types, \
    TS_FUNCTION_EXTRACTORS, TS_CLASSES_EXTRACTORS
from buckwheat.pipeline.input import LanguageClassifiedDirectory, transform_directory_to_files


def extract_identifiers_from_files(files_bag: bag.Bag) -> bag.Bag:
    """
    Extract identifiers with file-level granularity

    :param files_bag: dask bag with LanguageClassifiedFile items
    :return: dask Bag with Tuple[LanguageClassifiedFile, List[Entity]] type
    """
    identifiers_bag = files_bag.map(identifiers_extractor)
    return bag.zip(files_bag, identifiers_bag)


T = TypeVar("T")


def extract_identifiers_from_projects(projects_bag: bag.Bag) -> bag.Bag:
    """
    Extract identifiers with project-level granularity.
    Projects in result are represented using project path for now.

    :param projects_bag: dask bag with LanguageClassifiedDirectory items
    :return: dask bag with Tuple[str, List[Entity]] items
    """
    def combine(left: Tuple[T, ...], right: Tuple[T, ...]) -> Tuple[T, ...]:
        """Combine two tuples into new one containing all elements from both"""
        return left + right

    def binop(total: Tuple[BaseEntity, ...], x: Tuple[LanguageClassifiedDirectory, BaseEntity])\
            -> Tuple[BaseEntity, ...]:
        """Combine total with new element. Only entities are collected in total."""
        return total + (x[1],)

    def key_func(x: Tuple[LanguageClassifiedDirectory, BaseEntity]) -> str:
        """Return key from directory-entities tuples. Path of directory is used as key, as it should be unique."""
        return x[0].path

    def join_project_to_files(project_files: Tuple[LanguageClassifiedDirectory, List[LanguageClassifiedDirectory]]):
        """Return tuple of project instances for every file from the project."""
        project, files = project_files
        return tuple(project for _ in files)

    def flatten_list_of_lists(lst: List[List[T]]) -> List[T]:
        """Flat list of lists into one large list"""
        return [item for sublist in lst for item in sublist]

    files_bag = projects_bag.map(transform_directory_to_files)
    flat_projects_bag = bag.zip(projects_bag, files_bag).map(join_project_to_files).flatten()
    flat_files_bag = files_bag.flatten()
    identifiers_bag = flat_files_bag.map(identifiers_extractor)
    return (
        bag.zip(flat_projects_bag, identifiers_bag)
           .foldby(key_func, binop, tuple(), combine, tuple())
           .map(lambda group: (group[0], flatten_list_of_lists(group[1])))
    )


def extract_group_of_identifiers_from_parent_types(files_bag: bag.Bag, extractors: List[TreeSitterExtractor])\
        -> bag.Bag:
    """
    Extract TreeEntity from files bag using given extractors. Identifiers are extracted as children.

    :param files_bag: dask bag with LanguageClassifiedFile items
    :param extractors: list of extractors used for extraction
    :return: dask bag with Tuple[LanguageClassifiedFile, List[Entities]] items
    """
    extractor = compose_extractors_with_childrens_of_types(extractors, MERGED_IDENTIFIERS)
    return bag.zip(files_bag, files_bag.map(extractor))


def extract_identifiers_for_classes(files_bag: bag.Bag) -> bag.Bag:
    """
    Extract identifiers with class-level granularity.

    :param files_bag: dask bag with LanguageClassifiedFile items
    :return: dask bag with Tuple[LanguageClassifiedFile, List[Entities]] items
    """
    return extract_group_of_identifiers_from_parent_types(files_bag, TS_CLASSES_EXTRACTORS)


def extract_identifiers_for_functions(files_bag: bag.Bag) -> bag.Bag:
    """
    Extract identifiers with function-level granularity.

    :param files_bag: dask bag with LanguageClassifiedFile items
    :return: dask bag with Tuple[LanguageClassifiedFile, List[Entities]] items
    """
    return extract_group_of_identifiers_from_parent_types(files_bag, TS_FUNCTION_EXTRACTORS)
