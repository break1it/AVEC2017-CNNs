import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/Sample-02-Valence-TestResult/'

    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=int, delimiter=',')

        matrix = numpy.zeros([2, 2])
        for sample in data:
            matrix[sample[0]][sample[1]] += 1
        print(matrix)
