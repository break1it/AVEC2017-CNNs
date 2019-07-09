import numpy
import os
from AMIGO.Tools import F1Score_Calculator

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result/Sample-02-Valence-TestResult/'

    F1List = []
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=int, delimiter=',')

        matrix = numpy.zeros([2, 2])
        for sample in data:
            matrix[sample[0]][sample[1]] += 1

        F1List.append(F1Score_Calculator(matrix=matrix))

        if filename == '0001.csv':
            print(matrix)
        # exit()
    print(numpy.argmax(F1List), max(F1List))
