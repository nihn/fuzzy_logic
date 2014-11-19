from matplotlib import pyplot

from fuzzy import FuzzySet


class FuzzyPlot():
    """
    Class for easy plotting FuzzySet objects
    """

    @staticmethod
    def plot(fuzzy, *args, **kwargs):
        if not 'label' in kwargs:
            kwargs['label'] = fuzzy.name
        if not isinstance(fuzzy, FuzzySet):
            raise Exception('Only FuzzySet can be draw.')
        pyplot.plot(fuzzy.fuzzy['x'], fuzzy.fuzzy['u'], *args, **kwargs)

    @staticmethod
    def show():
        pyplot.show()

    def __del__(self):
        pyplot.clf()
