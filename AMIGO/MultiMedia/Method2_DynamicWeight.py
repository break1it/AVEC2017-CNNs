import numpy
import os
from AMIGO.Model.Model_NN import NeuralNetwork
import tensorflow
from sklearn.preprocessing import scale
import multiprocessing as mp
import time


def Treatment():
    datapath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step3-Concat/'
    labelpath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step2-Label/'

    for sampleIndex in range(2, 40):
        if sampleIndex in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue
        for part in ['Valence', 'Arousal']:
            savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/Experiment-Result-BLSTM1Layer-Step4-Result/Sample-%02d-%s' % (
                sampleIndex, part)

            if os.path.exists(savepath): continue
            os.makedirs(savepath)
            os.makedirs(savepath + '-TestResult')

            trainData = numpy.genfromtxt(
                fname=os.path.join(datapath, 'Sample-%02d-%s-TrainData.csv' % (sampleIndex, part)), dtype=float,
                delimiter=',')
            testData = numpy.genfromtxt(
                fname=os.path.join(datapath, 'Sample-%02d-%s-TestData.csv' % (sampleIndex, part)), dtype=float,
                delimiter=',')
            trainLabel = numpy.genfromtxt(
                fname=os.path.join(labelpath,
                                   'Vector-00-Sample-%02d-%s-MiddleResult-Train-Label.csv' % (sampleIndex, part)),
                dtype=float, delimiter=',')
            testLabel = numpy.genfromtxt(
                fname=os.path.join(labelpath,
                                   'Vector-00-Sample-%02d-%s-MiddleResult-Test-Label.csv' % (sampleIndex, part)),
                dtype=float, delimiter=',')

            totalData = numpy.concatenate([trainData, testData], axis=0)
            totalData = scale(totalData)
            trainData = totalData[0:numpy.shape(trainData)[0]]
            testData = totalData[numpy.shape(trainData)[0]:]

            print(sampleIndex, part, numpy.shape(trainData), numpy.shape(testData), numpy.shape(trainLabel),
                  numpy.shape(testLabel))

            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = NeuralNetwork(trainData=trainData, trainLabel=trainLabel)
                for episode in range(100):
                    print('\nEpisode %d Total Loss = %f' % (
                        episode, classifier.Train(logName=os.path.join(savepath, 'Loss-%04d.csv' % episode))))
                    classifier.Save(os.path.join(savepath, 'Network-%04d' % episode))
                    classifier.Test(os.path.join(savepath + '-TestResult', 'Predict-%04d.csv' % episode),
                                    testData=testData, testLabel=testLabel)
            # exit()


if __name__ == '__main__':
    processList = []

    for index in range(2):
        process = mp.Process(target=Treatment)
        process.start()
        processList.append(process)
        time.sleep(5)

    for process in processList:
        process.join()
