import numpy
import os
from AMIGO.Tools import F1Score_Calculator, Precision_Calculator

if __name__ == '__main__':
    choosePart = 'Arousal'

    datapath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step1-SoftMax-Normalization/'
    labelpath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step2-Label/'

    for index in range(41):
        if not os.path.exists(os.path.join(
                labelpath, 'Vector-00-Sample-%02d-%s-MiddleResult-Test-Label.csv' % (index, choosePart))): continue
        testLabel = numpy.genfromtxt(
            fname=os.path.join(labelpath, 'Vector-00-Sample-%02d-%s-MiddleResult-Test-Label.csv' % (index, choosePart)),
            dtype=int, delimiter=',')
        # print(index, numpy.shape(testLabel))

        totalPredict = numpy.zeros([numpy.shape(testLabel)[0], numpy.shape(testLabel)[1]])
        for vector in range(17):
            predict = numpy.genfromtxt(
                fname=os.path.join(datapath, 'Vector-%02d-Sample-%02d-%s-MiddleResult' % (vector, index, choosePart),
                                   'Test-0000-Predict.csv'), dtype=float, delimiter=',')
            totalPredict += predict
        totalPredict = numpy.argmax(totalPredict, axis=1)
        testLabel = numpy.argmax(testLabel, axis=1)

        matrix = numpy.zeros([2, 2])
        for index in range(len(testLabel)):
            matrix[testLabel[index]][totalPredict[index]] += 1

        print(F1Score_Calculator(matrix))
        # exit()
