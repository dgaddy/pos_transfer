import itertools
import numpy

def load(filename):
    f = open(filename, 'r')

    result = {}
    size = 0
    first = True
    for line in f:
        if first:
            first = False
            continue
        split = line.split()
        word = split[0]
        vect = numpy.array([float(x) for x in itertools.islice(split, 1, None)])
        result[word] = vect
        size = len(vect)

    return result, size
