"""
The functionality of the temporal slicing of projects.
"""
import datetime
import os
from typing import List


def get_dates(n_dates: int, day_delta: int, start_date: str = None) -> List[datetime.datetime]:
    """
    Creates a list of a given number of the datetime objects with a given step, starting
    from the start_date and going down. By default the start_date is the moment of calling.
    :param n_dates: number of dates.
    :param day_delta: the number of days between dates.
    :param start_date: the starting (latest) date of the slicing, in the format YYYY-MM-DD,
    the default value is the moment of calling.
    :return: a list of datetime objects.
    """
    dates = []
    if start_date is None:
        date = datetime.datetime.now()
    else:
        date = datetime.datetime.strptime(start_date,"%Y-%m-%d")
    for _ in range(n_dates):
        dates.append(date)
        date = date - datetime.timedelta(days=day_delta)
    dates.reverse()
    return dates


def checkout_by_date(repository: str, directory: str, before_date: datetime.datetime) -> None:
    """
    Checkout a given repository into a folder for a given date and time.
    :param repository: path to git repository.
    :param directory: path to target directory to store checkout from given repository.
    :param before_date: last commit before this date will be used for checkout.
    :return: None.
    """
    os.system("cp -r {repository} {directory}".format(repository=repository, directory=directory))
    os.system('cd {directory}; '
              'git checkout --quiet `git rev-list -n 1 --before="{date}" master` > /dev/null'
              .format(directory=directory, date=before_date.strftime("%Y-%m-%d")))
    # TODO: consider non-master branches
