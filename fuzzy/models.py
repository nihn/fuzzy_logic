import numpy as np

from fuzzy.base import Fuzzy
from fuzzy.constants import T, S, D, DECIMALS
from fuzzy.norms import MINMAX
from fuzzy.utils.decorators import require_fuzzy
from fuzzy.utils import utils


class FuzzySet(Fuzzy):

    def __init__(self, fuzzy, name='FuzzySet', norm=MINMAX):

        dt = np.dtype([('x', tuple), ('u', np.float)])

        super(FuzzySet, self).__init__(np.array(fuzzy, dtype=dt), name, norm)
        self.fuzzy.sort()

    @staticmethod
    def _get_intersect(set1, set2):
        return np.intersect1d(set1['x'], set2['x'], True)

    @staticmethod
    def _get_diffs(set1, set2):
        diff1 = np.setdiff1d(set1['x'], set2['x'], True)
        diff2 = np.setdiff1d(set2['x'], set1['x'], True)
        return diff1, diff2

    @staticmethod
    def _get_set_from_values(set, values):
        return set[set['x'].searchsorted(values)]

    @staticmethod
    def _get_values_setter(set1, set2):
        set1_copy = np.copy(set1)
        set2_copy = np.copy(set2)

        set1_copy['u'] = 0
        set2_copy['u'] = 0

        result = np.union1d(set1_copy, set2_copy)
        inter = np.intersect1d(set1_copy['x'], set2_copy['x'])

        def set_values(set):
            result['u'] = 0
            result[result['x'].searchsorted(set['x'])] = set

            if np.round(set[0]['u']) == 1 and set[0]['x'] != result[0]['x']:
                # TODO Why line below doesnt work?
                result[result['x'] < inter[0]]['u'] = 1
                for i, e in enumerate(result['x'] < inter[0]):
                    if not e:
                        break
                    result[i]['u'] = 1

            if np.round(set[-1]['u']) == 1 and set[-1]['x'] != result[-1]['x']:
                cond = result['x'] > inter[-1]
                for i, _ in enumerate(cond):
                    if not cond[-i-1]:
                        break
                    result[-i-1]['u'] = 1
            return result
        return set_values

    @staticmethod
    def _get_unions(set1, set2):
        """
        Method used to make two FuzzySets equal length and
        fill them with right values
        """
        setter = FuzzySet._get_values_setter(set1, set2)

        result1 = np.copy(setter(set1))
        result2 = setter(set2)

        return np.sort(result1), np.sort(result2)

    def _and_or(self, norm, set1, set2):
        """
        Call and or or based on norm value
        :param norm: T or S
        :param set1: set1
        :param set2: set2
        :return: new FuzzySet (set1 & set2) or (set1 | set2)
        """
        union1, union2 = self._get_unions(set1.fuzzy, set2.fuzzy)

        return FuzzySet(self.norm.eval(norm, union1, union2),
                        name='result %s %s %s' % (set1.name, norm, set2.name),
                        norm=set1.norm)

    @require_fuzzy
    def __or__(self, other):
        """
        | operator, use T norm
        :param other:
        :return: new FuzzySet (set1 | set2)
        """

        return self._and_or(T, self, other)

    @require_fuzzy
    def __and__(self, other):
        """
        & operator, use S norm
        :param other:
        :return: new FuzzySet (set1 & set2)
        """
        return self._and_or(S, self, other)

    def __invert__(self):

        return -self

    @require_fuzzy
    def __div__(self, other):

        return self.norm.eval(D, self, other)

    @require_fuzzy
    def __mul__(self, other):
        """
        Cartezian mul two sets, if second is FuzzyRel then call _mul_rel
        :param other:
        :return: new FuzzySet (set1 * set2)
        """

        if isinstance(other, FuzzyRel):
            return self._mul_rel(other)

        result_x = map(tuple, utils.cartesian((self.fuzzy['x'],
                                               other.fuzzy['x'])))
        result_u = utils.cartesian((self.fuzzy['u'], other.fuzzy['u']))

        set1 = FuzzySet(zip(result_x, result_u[:, 0]))
        set2 = FuzzySet(zip(result_x, result_u[:, 1]))

        result = self.norm.eval(T, set1, set2)
        return FuzzySet(map(tuple, result),
                        name='%s * %s' % (self.name, other.name),
                        norm=self.norm)

    def _mul_rel(self, other):
        """
        Mul set with relation
        :param other:
        :return: new FuzzyRel (set o rel)
        """
        if len(self) != other.fuzzy.shape[0]:
            # TODO valid exception
            raise Exception
        result = []
        for row in other.fuzzy.transpose():
            sub_result = []
            for index, u in enumerate(row):
                sub_result.append(self.norm.t(u, self[index]['u']))
            result.append(max(sub_result))

        return FuzzyRel(result, norm=self.norm, name='%s * %s' % (self.name,
                                                                  other.name))

    def core(self):

        return utils.core(self.fuzzy)

    def sup(self):

        return utils.sup(self.fuzzy)

    def crossover(self):

        return utils.crossover(self.fuzzy)

    def width(self):

        return utils.width(self.fuzzy)

    def con(self):

        self.fuzzy = utils.con(self.fuzzy)

    def dil(self):

        self.fuzzy = utils.dil(self.fuzzy)

    def int(self, beta=2):

        self.fuzzy = utils.int(self.fuzzy, beta)

    def dim(self, beta=2):

        self.fuzzy = utils.dil(self.fuzzy, beta)

    def alfa_cut(self, alfa):

        return utils.alfa_cut(self.fuzzy, alfa)

    def extension(self, function, *args, **kwargs):

        self.fuzzy = utils.extension(self.fuzzy, function, *args, **kwargs)


class FuzzyRel(Fuzzy):

    def __init__(self, fuzzy, name='FuzzyRel', norm=MINMAX):

        super(FuzzyRel, self).__init__(np.array(fuzzy), name, norm)

    @require_fuzzy
    def __mul__(self, other):
        """
        self o other
        :param other:
        :return: new FuzzyRel (rel1 o rel2)
        """
        if isinstance(other, FuzzySet):
            if len(other) != self.fuzzy.shape[1]:
                # TODO valid exception
                raise Exception
            result = []
            for row in self:
                sub_result = []
                for index, u in enumerate(row):
                    sub_result.append(self.norm.t(u, other[index]['u']))
                result.append(max(sub_result))

        elif isinstance(other, FuzzyRel):
            if self.fuzzy.shape[0] != other.fuzzy.shape[1]:
                # TODO valid exception
                raise Exception
            result = []
            for row in self:
                sub_result = []
                for index1, _ in enumerate(self):
                    sub2_result = []
                    for index2, u in enumerate(row):
                        sub2_result.append(self.norm.t(u, other[index2,
                                                                index1]))
                    sub_result.append(max(sub2_result))
                result.append(sub_result)
        else:
            raise NotImplemented

        return FuzzyRel(result, norm=self.norm,
                        name='%s * %s' % (self.name, other.name))

    @require_fuzzy
    def imp(self, other, norm=None):
        """
        self => other
        :param other: one dimension relation
        :param norm: one dimension relation
        :return: new FuzzyRel (rel1 => rel2)
        """

        if len(self.fuzzy.shape) > 1 or len(other.fuzzy.shape) > 1:
            raise Exception('Only one dimension fuzzy can use implication')

        if norm is None:
            norm = self.norm
        neg = -self
        result = []

        for u1 in other.fuzzy:
            sub_result = []
            for index, u2 in enumerate(self.fuzzy):
                sub_result.append(norm.s(neg[index], norm.t(u1, u2)))
            result.append(sub_result)

        return FuzzyRel(result, norm=norm, name='%s => %s' % (self.name,
                                                              other.name))
