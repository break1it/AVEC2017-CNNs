import tensorflow
import numpy
import os
from Auxiliary.Loader import Loader_CNN
from Model.HACN import HACN
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = Loader_CNN(maxSentence=5)

    attention = RNN_LocalAttentionInitializer
    attentionName = 'LA'
    attentionScope = 1

    savepath = 'D:/PythonProjects_Data/Experiment/HACN-%s' % attentionName
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')
    classifier = HACN(trainData=numpy.concatenate([trainData, developData], axis=0),
                      trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0), attention=attention,
                      attentionName=attentionName, attentionScope=attentionScope)
    for episode in range(100):
        print('\nEpisode %d Total Loss = %f' %
              (episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(logName=savepath + '-TestResult/%04d.csv' % episode, testData=testData, testLabel=testLabel)
