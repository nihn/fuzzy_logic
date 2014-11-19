import numpy as np

from fuzzy.constants import T, S, D, N


class Norm(object):
    """
    Base class for all norms
    """

    def eval(self, norm, set1, set2):
        """
        Function to evaluate Norm on two sets
        :param norm: {S, T, D, N}
        :param set1:
        :param set2:
        :return: set1 norm set2
        """

        if norm in (S, T, D):
            u1 = set1['u']
            u2 = set2['u']

            if norm == T:
                set1['u'] = self.t(u1, u2)
            elif norm == S:
                set1['u'] = self.s(u1, u2)
            elif norm == D:
                set1['u'] = self.d(u1, u2)
            return set1
        elif norm == N:
            try:
                return self.neg(set1['u'])
            except IndexError:
                return self.neg(set1)
        else:
            raise NotImplemented

    def __repr__(self):
        return 'norm=%s' % type(self).__name__

    @staticmethod
    def t(u1, u2):
        raise NotImplemented

    @staticmethod
    def s(u1, u2):
        raise NotImplemented

    @staticmethod
    def d(u1, u2):
        raise NotImplemented

    @staticmethod
    def neg(u):
        raise NotImplemented


class MinMax(Norm):

    @staticmethod
    def t(u1, u2):
        return np.minimum(u1, u2)

    @staticmethod
    def s(u1, u2):
        return np.maximum(u1, u2)

    @staticmethod
    def d(u1, u2):
        return np.minimum(u1, 1-u2)

    @staticmethod
    def neg(u):
        return 1-u


class Lukasiewicz(Norm):

    @staticmethod
    def t(u1, u2):
        return np.minimum(u1 + u2 - 1, 0)

    @staticmethod
    def s(u1, u2):
        return np.maximum(u1 + u2, 1)


class AlgProb(Norm):

    @staticmethod
    def t(u1, u2):
        return u1 * u2

    @staticmethod
    def s(u1, u2):
        return u1 + u2 - u1 * u2

    @staticmethod
    def d(u1, u2):
        return u1 * (1 - u2)


class Drastic(Norm):

    @staticmethod
    def _get_maxes_and_mins(u1, u2):
        return np.maximum(u1, u2), np.minimum(u1, u2)

    @staticmethod
    def t(u1, u2):

        maxes, mins = Drastic._get_maxes_and_mins(u1, u2)
        cond = maxes == 1
        return np.where(cond, mins[cond], 0)

    @staticmethod
    def s(u1, u2):

        maxes, mins = Drastic._get_maxes_and_mins(u1, u2)
        cond = mins == 0
        return np.where(cond, maxes[cond], 1)

    @staticmethod
    def d(u1, u2):
        if u1 == 1:
            return 1 - u2
        elif u2 > 0:
            return 0
        else:
            return u1


class Fodor(Norm):

    @staticmethod
    def t(u1, u2):

        mins = np.minimum(u1, u2)
        cond = u1 + u2 > 1
        return np.where(cond, mins[cond], 0)

    @staticmethod
    def s(u1, u2):

        maxes = np.maximum(u1, u2)
        cond = u1 + u2 < 1
        return np.where(cond, maxes[cond], 1)


class Einstein(Norm):

    @staticmethod
    def t(u1, u2):

        return u1*u2 / (2 - u1 + u2 - u1*u2)

    @staticmethod
    def s(u1, u2):

        return (u1 + u2) / (1 + u1*u2)

    @staticmethod
    def neg(u):
        return np.sqrt(1-u*u)


class Yager(Norm):

    w = 1
    y = 1

    @staticmethod
    def t(u1, u2):

        temp = ((1 - u1) ** Yager.w + (1 - u2) ** Yager.w) ** (1 / Yager.w)
        return 1 - np.where(temp > 1, 1, temp)

    @staticmethod
    def s(u1, u2):

        temp = (u1 ** Yager.w + u2 ** Yager.w) ** (1 / Yager.w)
        return np.where(temp > 1, 1, temp)

    @staticmethod
    def neg(u):

        return (1 - u ** Yager.y) ** (1 / Yager.y)


MINMAX = MinMax()
ALGPROB = AlgProb()
DRASTIC = Drastic()
FODOR = Fodor()
EINSTEIN = Einstein()
YAGER = Yager()
