from fuzzy import FuzzySet, FuzzyPlot, FuzzyRel
from fuzzy.utils import *
from fuzzy.norms import EINSTEIN

set1 = FuzzySet(gauss(1, 8), norm=EINSTEIN)
set2 = FuzzySet(sigmoid(1, 0.5))

set3 = FuzzySet(linear(1, 5, reverse=True))
set4 = FuzzySet(trapezoid(-3, 2, 5, 7))

p1 = FuzzyPlot()
p2 = FuzzyPlot()

p1.plot(set1)
p1.plot(set2)
p1.plot(set1 & set2)
p1.plot(set1 | set2)
p1.show()

p2.plot(set3)
p2.plot(set4)
p2.plot(set3 & set4)
p2.plot(set3 | set4)
p2.show()

s1 = FuzzyRel([(0.2, 0.5, 1), (0.3, 0.2, 0.1)])
s2 = FuzzySet([(1, 0.7), (2, 0.6)])
print s1
print s2
print s2 * s1

s3 = FuzzyRel([0.1, 0.2, 0.6])
s4 = FuzzyRel([0.4, 0.7, 1])

print s3
print s4
print s3.imp(s4)

