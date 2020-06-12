"""
Auxiliary functionality.
"""
import os
import subprocess
from typing import Any, List


class RepositoryError(ValueError):
    """
    A special error for catching wrong links to repositories and skipping such repositories.
    """

    def __init__(self, *args):
        ValueError.__init__(self, *args)


def read_file(file: str) -> str:
    """
    Read the contents of the file.
    :param file: the path to the file.
    :return: the contents of the file.
    """
    with open(file) as fin:
        return fin.read()


def split_list_into_batches(lst: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a given list into sublists with a given maximum number of items.
    :param lst: a list.
    :param batch_size: the maximum number of items in the sublists.
    :return: a list of lists, splitting the original list into batches.
    """
    return [lst[x:x + batch_size] for x in range(0, len(lst), batch_size)]


def assert_trailing_slash(link: str) -> str:
    """
    Add a trailing slash to a link if there isn't one.
    :param link: link to directory or Web site.
    :return: the same link with a trailing slash.
    """
    link = link.rstrip()
    if link[-1] == "/":
        return link
    else:
        return link + "/"


def clone_repository(repository: str, directory: str) -> None:
    """
    Clone a given repository into a folder.
    :param repository: a link to GitHub repository, either HTTP or HTTPs.
    :param directory: path to target directory to clone the repository.
    :return: none.
    """
    if "://" in repository:
        body = repository.split("://")[1]
    else:
        raise RepositoryError(f"{repository} is not a valid link!")
    repository = "https://user:password@" + body
    os.system(f"git clone --quiet --depth 1 {repository} {directory}")


def get_latest_commit(directory: str) -> str:
    """
    Get the current commit hash from the Git directory.
    :param directory: the path to a Git directory.
    :return: commit hash.
    """
    command = f"cd {directory}; git rev-parse HEAD"
    return subprocess.check_output(command, shell=True, text=True).rstrip()


def get_full_path(file: str, directory: str) -> str:
    """
    Get the full path to file from the full path to a directory and a relative path to that
    file in that directory.
    :param file: the relative path to file in a directory.
    :param directory: the full path of a directory.
    :return: the full path to file.
    """
    return os.path.abspath(os.path.join(directory, file))
