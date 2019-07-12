import numpy
import os
import tensorflow
from AMIGO.Loader import Loader
from AMIGO.Tools import LabelPretreatment
from AMIGO.Model.Model_CNN import CNN
import multiprocessing as mp
import time


def Treatment():
    layers = 3
    for vector in range(17):
        for appoint in range(2, 41):
            if appoint in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue
            for axis in range(2):
                graph = tensorflow.Graph()
                with graph.as_default():
                    savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result-CNN1Conv-Layer%02d/Vector-%02d-Sample-%02d-%s' % (
                        layers, vector, appoint, ['Valence', 'Arousal'][axis])
                    if os.path.exists(savepath): continue

                    os.makedirs(savepath)
                    os.makedirs(savepath + '-TestResult')
                    trainData, trainLabel, testData, testLabel = Loader(name='Vector%02d' % vector, appoint=appoint)
                    trainData = numpy.reshape(trainData, [-1, 20 * 65])
                    trainLabel = LabelPretreatment(trainLabel[:, axis])
                    testData = numpy.reshape(testData, [-1, 20 * 65])
                    testLabel = LabelPretreatment(testLabel[:, axis])

                    classifier = CNN(trainData=trainData, trainLabel=trainLabel, batchSize=64, learningRate=1E-3,
                                     layers=layers)
                    for episode in range(10):
                        print('\nEpisode %d Total Loss = %f' % (
                            episode, classifier.Train(logName=os.path.join(savepath, 'Loss-%04d.csv' % episode))))
                        classifier.Test(logName=os.path.join(savepath + '-TestResult', '%04d.csv' % episode),
                                        testData=testData, testLabel=testLabel)


if __name__ == '__main__':
    processList = []

    for index in range(2):
        process = mp.Process(target=Treatment)
        process.start()
        processList.append(process)
        time.sleep(5)

    for process in processList:
        process.join()
