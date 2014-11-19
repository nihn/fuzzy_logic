from functools import wraps

from fuzzy.base import Fuzzy
from fuzzy.exceptions import FuzzyError


def require_fuzzy(function):
    """
    only fuzzy operations allowed
    :param function:
    :return:
    """
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not isinstance(args[1], Fuzzy):
            raise FuzzyError('Operation allowed only on Fuzzy instantions')
        return function(*args, **kwargs)
    return wrapper
