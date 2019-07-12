import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step1-SoftMax-Normalization/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step3-Concat/'

    # os.makedirs(savepath)
    choosePart = 'Arousal'
    for index in range(2, 41):
        trainData, testData = [], []
        for vector in range(17):
            if not os.path.exists(
                    os.path.join(loadpath, 'Vector-%02d-Sample-%02d-%s-MiddleResult' % (vector, index, choosePart),
                                 'Train-0000-Predict.csv')): continue
            partData = numpy.genfromtxt(
                fname=os.path.join(loadpath, 'Vector-%02d-Sample-%02d-%s-MiddleResult' % (vector, index, choosePart),
                                   'Train-0000-Predict.csv'), dtype=float, delimiter=',')
            if len(trainData) == 0:
                trainData = partData
            else:
                trainData = numpy.concatenate([trainData, partData], axis=1)

            partData = numpy.genfromtxt(
                fname=os.path.join(loadpath, 'Vector-%02d-Sample-%02d-%s-MiddleResult' % (vector, index, choosePart),
                                   'Test-0000-Predict.csv'), dtype=float, delimiter=',')
            if len(testData) == 0:
                testData = partData
            else:
                testData = numpy.concatenate([testData, partData], axis=1)

            print(index, vector, numpy.shape(trainData), numpy.shape(testData))

        if len(trainData) != 0:
            with open(os.path.join(savepath, 'Sample-%02d-%s-TrainData.csv' % (index, choosePart)), 'w') as file:
                for indexX in range(numpy.shape(trainData)[0]):
                    for indexY in range(numpy.shape(trainData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(trainData[indexX][indexY]))
                    file.write('\n')
            with open(os.path.join(savepath, 'Sample-%02d-%s-TestData.csv' % (index, choosePart)), 'w') as file:
                for indexX in range(numpy.shape(testData)[0]):
                    for indexY in range(numpy.shape(testData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(testData[indexX][indexY]))
                    file.write('\n')
        # exit()
