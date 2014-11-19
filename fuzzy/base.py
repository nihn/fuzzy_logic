from fuzzy.exceptions import FuzzyError
from fuzzy.norms import Norm
from fuzzy.constants import N


class Fuzzy(object):
    """
    base class for Fuzzy related objects
    """

    def __init__(self, fuzzy, name='Fuzzy', norm=Norm()):
        if not isinstance(norm, Norm):
            raise FuzzyError('Invalid Norm %s' % norm)

        self.fuzzy = fuzzy
        self.norm = norm
        self.name = name

    def __repr__(self):
        return '%s\n%s' % (repr(self.fuzzy), repr(self.norm))

    def __str__(self):
        return str(self.fuzzy)

    def __getitem__(self, item):
        return self.fuzzy[item]

    def __setitem__(self, key, value):
        self.fuzzy[key] = value

    def __len__(self):
        return len(self.fuzzy)

    def __neg__(self):
        return self.norm.eval(N, self.fuzzy, None)
