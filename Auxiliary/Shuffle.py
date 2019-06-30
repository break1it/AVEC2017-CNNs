import numpy
import random


def Shuffle_Double(a, b):
    if len(a) != len(b):
        raise RuntimeError("Input Don't Have Same Len.")

    index = numpy.arange(0, len(a))
    random.shuffle(index)
    newA, newB = [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
    return newA, newB


def Shuffle_Triple(a, b, c):
    if len(a) != len(b) or len(b) != len(c):
        raise RuntimeError("Input Don't Have Same Len.")

    index = numpy.arange(0, len(a))
    random.shuffle(index)
    newA, newB, newC = [], [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
        newC.append(c[sample])
    return newA, newB, newC


def Shuffle_Fourfold(a, b, c, d):
    if len(a) != len(b) or len(b) != len(c) or len(c) != len(d):
        raise RuntimeError("Input Don't Have Same Len.")

    index = numpy.arange(0, len(a))
    random.shuffle(index)
    newA, newB, newC, newD = [], [], [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
        newC.append(c[sample])
        newD.append(d[sample])
    return newA, newB, newC, newD


def Shuff_Fivefold(a, b, c, d, e):
    if len(a) != len(b) or len(b) != len(c) or len(c) != len(d) or len(d) != len(e):
        # print(len(a), len(b), len(c), len(d), len(e))
        raise RuntimeError("Input Don't Have Same Len.")

    index = numpy.arange(0, len(a))
    random.shuffle(index)
    newA, newB, newC, newD, newE = [], [], [], [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
        newC.append(c[sample])
        newD.append(d[sample])
        newE.append(e[sample])
    return newA, newB, newC, newD, newE


if __name__ == '__main__':
    # Test
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2]
    print(Shuffle_Double(a=x, b=y))
