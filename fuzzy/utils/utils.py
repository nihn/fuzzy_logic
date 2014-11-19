"""
FuzzySet utilities functions
"""
import numpy as np


def _find_closest(array, value):
    return np.abs(array['u']-value).argmin()


def core(array):

    return array[array['u'] == 1]['x']


def sup(array):

    return array[array['u'] > 0]['x']


def crossover(array):

    idx1 = _find_closest(array, 0.5)
    idx2 = _find_closest(array[array != array[idx1]], 0.5)
    if abs(idx1 - idx2) == 1:
        return None, None
    return array[idx1], array[idx2]


def width(array):

    x1, x2 = crossover(array)
    if x1 is not None:
        return abs(x2 - x1)
    return 0


def con(array):
    """
    concentration
    :param array:
    :return: concetrated array
    """
    temp = np.array(array)
    temp['u'] = temp['u'] * temp['u']
    return temp


def dil(array):

    temp = np.array(array)
    temp['u'] = np.sqrt(temp['u'])
    return temp


def int(array, beta=2):
    """
    increase contrast
    :param array:
    :param beta:
    :return:
    """
    temp = np.array(array)
    cond = temp['u'] < 0.5
    inv_cond = np.invert(cond)

    temp['u'][cond] = 2 ** (beta - 1) * \
        temp['u'][cond] ** beta
    temp['u'][inv_cond] = 1 - 2 ** (beta - 1) * \
        (1 - temp['u'][inv_cond]) ** beta

    return temp


def dim(array, beta=2):
    """
    decrease contrast
    :param array:
    :param beta:
    :return:
    """
    temp = np.array(array)
    cond = temp['u'] >= 0.5
    inv_cond = temp['u'] = np.invert(cond)

    temp['u'][cond] = np.sqrt(temp['u'][cond] /
                              2 ** (beta - 1), beta)
    temp['u'][inv_cond] = \
        1 - np.sqrt((1 - temp['u'][inv_cond]) / 2 ** (beta - 1),
                    beta)

    return temp


def alfa_cut(array, alfa):

    cond = array['u'] >= alfa
    return np.where(cond, 1, 0)


def extension(array, function, *args, **kwargs):

    temp = function(array, *args, **kwargs)
    temp.sort()[::-1]

    return temp[np.unique(temp['x'], return_index=True)[1]]


def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out
