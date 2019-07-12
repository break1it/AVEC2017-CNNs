import numpy
from AMIGO.Tools import F1Score_Calculator, Precision_Calculator

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-NN-Step4-Result/Sample-%02d-Arousal-TestResult/Predict-%04d.csv'

    for sampleIndex in range(2, 41):
        if sampleIndex in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue
        F1List, PrecisionList = [], []
        for index in range(100):
            data = numpy.genfromtxt(fname=loadpath % (sampleIndex, index), dtype=int, delimiter=',')

            matrix = numpy.zeros([2, 2])
            for index in range(len(data)):
                matrix[data[index][0]][data[index][1]] += 1

            F1List.append(F1Score_Calculator(matrix=matrix))
            PrecisionList.append(Precision_Calculator(matrix=matrix))
        print(max(F1List), '\t', max(PrecisionList))
