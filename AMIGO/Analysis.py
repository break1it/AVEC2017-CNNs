import numpy
import os
from AMIGO.Tools import F1Score_Calculator

if __name__ == '__main__':
    for index in range(41):
        loadpath = 'D:/PythonProjects_Data/AMIGO/ExperimentResult/BLSTM2Layer/Vector-05-Sample-%02d-Arousal-TestResult/' % index
        if not os.path.exists(loadpath): continue
        F1List, matrixList = [], []
        for filename in os.listdir(loadpath):
            if not os.path.exists(os.path.join(loadpath, filename)): continue
            data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=int, delimiter=',')

            matrix = numpy.zeros([2, 2])
            for sample in data:
                matrix[sample[0]][sample[1]] += 1

            F1List.append(F1Score_Calculator(matrix=matrix))
            matrixList.append(matrix)

            # if filename == '0001.csv':
            #     print(matrix)
            # exit()
        # print(max(F1List), numpy.argmax(F1List))
        print(max(F1List))
