import numpy as np

from functools import partial

from fuzzy.constants import FUN_DTYPE, DX, DECIMALS


array_dtype = partial(np.array, dtype=FUN_DTYPE)


def _get_arguments(args, dx=DX):
    """
    Get x arguments
    :param args:
    :param dx:
    :return:
    """
    args = [(args[i-1], arg) for i, arg in enumerate(sorted(args))][1:]

    axes = [np.around(np.arange(args[0][0], args[0][1] + dx, dx),
                      decimals=DECIMALS)]
    for arg in args[1:-1]:
        axes.append(np.around(np.arange(arg[0] + dx, arg[1] + dx, dx),
                              decimals=DECIMALS))
    axes.append(np.around(np.arange(args[-1][0] + dx, args[-1][1] + dx, dx),
                          decimals=DECIMALS))

    return axes


def linear(start, stop, dx=DX, reverse=False):
    """
    Linear function
    :param start: for x < start u = 0 or 1
    :param stop: for x > start u = 1 or 1
    :param dx:
    :param reverse: if True 'a' function parameter = -1
    :return: array for FuzzySet
    """
    x = _get_arguments((start, stop), dx)[0]
    y = stop - x if reverse else x - start
    return array_dtype(zip(x, y/(stop-start)))


def triangle(start, middle, stop, dx=DX):
    """
    Triangle function
    :param start: for x < start u = 0
    :param middle: for x == middle u = 0
    :param stop: for x > start u = 0
    :param dx:
    :return: array for FuzzySet
    """
    x1, x2 = _get_arguments((start, middle, stop), dx)

    y1 = array_dtype(zip(x1, (x1-start)/(middle-start)))
    y2 = array_dtype(zip(x2, (stop-x2)/(stop-middle)))

    return np.concatenate((y1, y2))


def trapezoid(start, middle1, middle2, stop, dx=DX):
    """
    Trapezoid function
    :param start: for x < start u = 0
    :param middle1: for x > middle1 and < middle2 = 1
    :param middle2: for x > middle1 and < middle2 = 1
    :param stop: for x > stop u = 0
    :param dx:
    :return: array for FuzzySet
    """
    x1, x2, x3, = _get_arguments((start, middle1, middle2, stop), dx)

    y1 = array_dtype(zip(x1, (x1-start)/(middle1-start)))
    y2 = array_dtype(zip(x2, np.ones(len(x2))))
    y3 = array_dtype(zip(x3, (stop-x3)/(stop-middle2)))

    return np.concatenate((y1, y2, y3))


def s(start, stop, dx=DX):
    """
    S function (square)
    :param start: for x < start u = 1
    :param stop: for x > start u = 0
    :param dx:
    :return: array for FuzzySet
    """
    x1, x2 = _get_arguments((start, (start+stop)/2, stop), dx)

    y1 = array_dtype(zip(x1, 2*((x1-start)/(stop-start))**2))
    y2 = array_dtype(zip(x2, 1 - 2*((x2-stop)/(stop-start))**2))

    return np.concatenate((y1, y2))


def z(start, stop, dx=DX):
    """
    z function (square)
    :param start: for x < start u = 0
    :param stop: for x > start u = 1
    :param dx:
    :return: array for FuzzySet
    """
    temp = s(start, stop, dx)
    temp['u'] = 1 - temp['u']
    return temp


def pi(start, middle, stop, dx=DX):
    """
    pi function
    :param start:
    :param middle:
    :param stop:
    :param dx:
    :return: array for FuzzySet
    """
    return np.concatenate((s(start, middle, dx), z(middle, stop, dx)))


def gauss(middle, sigma, dx=DX):
    """
    gauss function
    :param middle:
    :param sigma:
    :param dx:
    :return: array for FuzzySet
    """
    radius = sigma * 5
    x = _get_arguments((middle-radius, middle+radius), dx)[0]

    return array_dtype(zip(x, np.exp(-(x-middle)**2/(2*sigma**2))))


def sigmoid(middle, beta, dx=DX):
    """
    sigmoid function
    :param middle:
    :param beta:
    :param dx:
    :return: array for FuzzySet
    """
    radius = 10 / beta
    x = _get_arguments((middle-radius, middle+radius), dx)[0]

    return array_dtype(zip(x, 1/(1 + np.exp(-beta*(x-middle)))))
