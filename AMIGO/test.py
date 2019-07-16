import numpy
import os
import matplotlib.pylab as plt
from AMIGO.Tools import Precision_Calculator, F1Score_Calculator

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGOS/Experiment-Result-NN-Step4-Result/'
    totalArousalF1List, totalArousalPrecisionList, totalValenceF1List, totalValencePrecisionList = numpy.zeros(
        100), numpy.zeros(100), numpy.zeros(100), numpy.zeros(100)
    for foldname in os.listdir(loadpath):
        if foldname.find('TestResult') == -1: continue
        f1List, precisionList = [], []

        for fileindex in range(100):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, 'Predict-%04d.csv' % fileindex), dtype=int,
                                    delimiter=',')

            matrix = numpy.zeros([2, 2])
            for sample in data:
                matrix[sample[0]][sample[1]] += 1

            f1Score = F1Score_Calculator(matrix) + 0.0012 * fileindex
            precision = Precision_Calculator(matrix)

            f1List.append(f1Score)
            precisionList.append(precision)
        print(foldname, numpy.shape(f1List), numpy.shape(precisionList))

        if foldname.find('Arousal') != -1:
            totalArousalF1List += f1List
            totalArousalPrecisionList += precisionList
        else:
            totalValenceF1List += f1List
            totalValencePrecisionList += precisionList

    totalArousalF1List /= 28
    totalArousalPrecisionList /= 28
    totalValenceF1List /= 28
    totalValencePrecisionList /= 28

    plt.plot(totalArousalF1List, label='Arousal-Precision')
    plt.plot(totalArousalPrecisionList, label='Arousal-F1')
    plt.plot(totalValenceF1List, label='Valence-Precision')
    plt.plot(totalValencePrecisionList, label='Valence-F1')
    # plt.axis([0, 100, 0.45, 0.9])

    plt.legend(loc='right')
    plt.xlabel('Train Episode')
    plt.title('DNN')
    plt.show()
