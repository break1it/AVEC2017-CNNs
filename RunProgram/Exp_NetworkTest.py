import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.SimpleCNN import SimpleCNN_3Layers, SimpleCNN_2Layers, SimpleCNN_2_5Layers
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer
from Model.AttentionMechanism.CNN_LocalAttention import CNN_LocalAttentionInitializer

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = \
        Loader_CNN(partName='CNN-10')

    attention = CNN_LocalAttentionInitializer
    attentionName = 'CLA'
    attentionScope = [3, 3]

    savepath = 'D:/PythonProjects_Data/Experiment/SimpleCNN-Layer2_5-%s' % attentionName
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    classifier = SimpleCNN_2_5Layers(
        trainData=numpy.concatenate([trainData, developData], axis=0),
        trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0),
        attention=attention, attentionName=attentionName, attentionScope=attentionScope)
    # classifier.Valid()
    for episode in range(100):
        print('\nTrain Episode %d Total Loss = %f' % (
            episode, classifier.Train(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(logName=savepath + '-TestResult/%04d.csv' % episode, testData=testData, testLabel=testLabel)
