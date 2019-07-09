import os
import numpy
import tensorflow
from AMIGO.Loader import Loader
from AMIGO.Tools import LabelPretreatment
from AMIGO.Model.Model_RNN import BLSTM

if __name__ == '__main__':
    for appoint in range(2, 41):
        if appoint in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue
        for axis in range(2):
            graph = tensorflow.Graph()
            with graph.as_default():
                savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result/Sample-%02d-%s' % (
                    appoint, ['Valence', 'Arousal'][axis])
                if os.path.exists(savepath): continue

                os.makedirs(savepath)
                os.makedirs(savepath + '-TestResult')
                trainData, trainLabel, testData, testLabel = Loader(name='Vector00', appoint=appoint)
                trainLabel = LabelPretreatment(trainLabel[:, axis])
                testLabel = LabelPretreatment(testLabel[:, axis])

                classifier = BLSTM(trainData=trainData, trainLabel=trainLabel, learningRate=1E-4)
                # classifier.Train(logName='log.csv')
                for episode in range(100):
                    print('\nEpisode %d Total Loss = %f' % (
                        episode, classifier.Train(logName=savepath + '/%04d.csv' % episode)))
                    classifier.Test(logName=savepath + '-TestResult/%04d.csv' % episode, testData=testData,
                                    testLabel=testLabel)
