import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.CRNN import CRNN
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer
from Model.AttentionMechanism.CNN_LocalAttention import CNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = \
        Loader_CNN(partName='CNN-10', maxSentence=99999)

    firstAttention = CNN_LocalAttentionInitializer
    firstAttentionScope = [3, 3]
    firstAttentionName = 'CLA'
    secondAttention = RNN_StandardAttentionInitializer
    secondAttentionScope = None
    secondAttentionName = 'RSA'

    loadpath = 'D:/PythonProjects_Data/Experiment/CRNN-%s-Part/Network-0099' % firstAttentionName
    # savepath = 'D:/PythonProjects_Data/Experiment/CRNN-%s-%s-Total' % (firstAttentionName, secondAttentionName)
    # os.makedirs(savepath)
    # os.makedirs(savepath + '-TestResult')

    classifier = CRNN(
        trainData=numpy.concatenate([trainData, developData], axis=0),
        trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0),
        firstAttention=firstAttention, firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope,
        secondAttention=secondAttention, secondAttentionName=secondAttentionName,
        secondAttentionScope=secondAttentionScope)
    # classifier.Valid()
    classifier.Load(loadpath=loadpath)
    classifier.Valid()
    # classifier.Train_Part(logName='log.csv')
    # for episode in range(100):
    #     print('\nTrain Episode %d Total Loss = %f' % (
    #         episode, classifier.Train_Total(logName=savepath + '/Loss-%04d.csv' % episode)))
    #     classifier.Save(savepath=savepath + '/Network-%04d' % episode)
    #     classifier.Test_Total(logName=savepath + '-TestResult/%04d.csv' % episode, testData=testData,
    #                           testLabel=testLabel)
