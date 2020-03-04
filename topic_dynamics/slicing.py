"""
The functionality of the temporal slicing of projects.
"""

import os
from typing import List
from datetime import datetime, timedelta


def get_dates(number: int, delta: int) -> List:
    """
    Creates a list of a given number of the datetime objects with a given step.
    :param number: the amount of dates.
    :param delta: the time step between dates
    :return: a list of datetime objects.
    """
    dates = []
    date = datetime.now()
    for i in range(number):
        dates.append(date)
        date = date - timedelta(days=delta)
    dates.sort()
    return dates


def checkout_by_date(repository: str, directory: str, date: datetime) -> None:
    """
    Checkout a given repository into a folder for a given date and time.
    :param repository: address of processed project.
    :param directory: address of target directory for a checkout.
    :param date: date and time of the last commit for the checkout
    :return: None.
    """
    os.system('cp -r ' + repository + ' ' + directory)
    os.system('(cd ' + directory + '; git checkout --quiet `git rev-list -n 1 --before="'
              + date.strftime('%Y-%m-%d') + '" master` > /dev/null)')
    # TODO: consider non-master branches
