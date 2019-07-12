import os
import tensorflow
from AMIGO.Loader import Loader
from AMIGO.Tools import LabelPretreatment
from AMIGO.Model.Model_RNN import BLSTM_NoAttention
import multiprocessing as mp
import time


def Treatment():
    for vector in range(17):
        for appoint in range(2, 41):
            if appoint in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue
            for axis in range(2):
                graph = tensorflow.Graph()
                with graph.as_default():
                    savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result-BLSTM1Layer-NoAttention/Vector-%02d-Sample-%02d-%s' % (
                        vector, appoint, ['Valence', 'Arousal'][axis])
                    if os.path.exists(savepath): continue

                    os.makedirs(savepath)
                    os.makedirs(savepath + '-TestResult')
                    os.makedirs(savepath + '-MiddleResult')
                    trainData, trainLabel, testData, testLabel = Loader(name='Vector%02d' % vector, appoint=appoint)
                    trainLabel = LabelPretreatment(trainLabel[:, axis])
                    testLabel = LabelPretreatment(testLabel[:, axis])

                    classifier = BLSTM_NoAttention(trainData=trainData, trainLabel=trainLabel, learningRate=1E-4,
                                                   rnnLayers=1, batchSize=64)
                    # classifier.Train(logName='log.csv')
                    # classifier.Valid()
                    # exit()
                    for episode in range(10):
                        print('\nEpisode %d Total Loss = %f' % (
                            episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
                        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
                        classifier.Test(logName=savepath + '-TestResult/%04d.csv' % episode, testData=testData,
                                        testLabel=testLabel)


if __name__ == '__main__':
    processList = []

    for index in range(1):
        process = mp.Process(target=Treatment)
        process.start()
        processList.append(process)
        time.sleep(5)

    for process in processList:
        process.join()
