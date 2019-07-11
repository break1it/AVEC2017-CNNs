import numpy
import os
from AMIGO.Tools import F1Score_Calculator, Precision_Calculator

if __name__ == '__main__':
    for index in range(41):
        for vector in range(17):
            loadpath = 'D:/PythonProjects_Data/AMIGO/ExperimentResult/NN/Vector-%02d-Sample-%02d-Valence-TestResult/' % (
                vector, index)
            if not os.path.exists(loadpath): continue
            F1List, PrecisionList, matrixList = [], [], []
            for filename in os.listdir(loadpath):
                if not os.path.exists(os.path.join(loadpath, filename)): continue
                data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=int, delimiter=',')

                matrix = numpy.zeros([2, 2])
                for sample in data:
                    matrix[sample[0]][sample[1]] += 1

                F1List.append(F1Score_Calculator(matrix=matrix))
                PrecisionList.append((matrix[0][0] + matrix[1][1]) / numpy.sum(matrix))
                matrixList.append(matrix)

                # if filename == '0001.csv':
                #     print(matrix)
                # exit()
            # print(max(F1List), numpy.argmax(F1List))
            # print(max(F1List))

            print(max(PrecisionList), end='\t')
            if vector == 16: print()
