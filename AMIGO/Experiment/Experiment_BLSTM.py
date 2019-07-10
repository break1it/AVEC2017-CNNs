import os
import numpy
import tensorflow
from AMIGO.Loader import Loader
from AMIGO.Tools import LabelPretreatment, F1Score_Calculator
from AMIGO.Model.Model_RNN import BLSTM
import multiprocessing as mp
import time


def Treatment():
    for vector in range(17):
        for appoint in range(2, 41):
            if appoint in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue
            for axis in range(2):
                graph = tensorflow.Graph()
                with graph.as_default():
                    savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result-BLSTM1Layer/Vector-%02d-Sample-%02d-%s' % (
                        vector, appoint, ['Valence', 'Arousal'][axis])
                    if os.path.exists(savepath): continue

                    os.makedirs(savepath)
                    os.makedirs(savepath + '-TestResult')
                    os.makedirs(savepath + '-MiddleResult')
                    trainData, trainLabel, testData, testLabel = Loader(name='Vector%02d' % vector, appoint=appoint)
                    trainLabel = LabelPretreatment(trainLabel[:, axis])
                    testLabel = LabelPretreatment(testLabel[:, axis])

                    classifier = BLSTM(trainData=trainData, trainLabel=trainLabel, learningRate=1E-4, rnnLayers=1,
                                       batchSize=64)
                    # classifier.Train(logName='log.csv')

                    bestF1 = 0.0
                    for episode in range(10):
                        print('\nEpisode %d Total Loss = %f' % (
                            episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
                        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
                        classifier.Test(logName=savepath + '-TestResult/%04d.csv' % episode, testData=testData,
                                        testLabel=testLabel)

                        f1Data = numpy.genfromtxt(fname=savepath + '-TestResult/%04d.csv' % episode, dtype=int,
                                                  delimiter=',')
                        matrix = numpy.zeros([2, 2])
                        for sample in f1Data:
                            matrix[sample[0]][sample[1]] += 1

                        if bestF1 < F1Score_Calculator(matrix=matrix):
                            bestF1 = F1Score_Calculator(matrix=matrix)
                            classifier.MiddleResult(logName=savepath + '-MiddleResult/Train-%04d' % episode,
                                                    data=trainData, label=trainLabel)
                            classifier.MiddleResult(logName=savepath + '-MiddleResult/Test-%04d' % episode,
                                                    data=testData, label=testLabel)


if __name__ == '__main__':
    processList = []

    for index in range(2):
        process = mp.Process(target=Treatment)
        process.start()
        processList.append(process)
        time.sleep(5)

    for process in processList:
        process.join()
