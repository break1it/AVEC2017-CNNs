import os
import numpy
import tensorflow
from Auxiliary.Loader import Loader_CNN_Flexible
from Model.CRNN_Mask import CRNN_Mask
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer_Mask

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_CNN_Flexible(partName='CNN-10', maxSentence=10)

    # print(trainSeq[0])

    firstAttention = CNN_StandardAttentionInitializer_Mask
    firstAttentionScope = None
    firstAttentionName = 'CSA'

    savepath = 'D:/PythonProjects_Data/Experiment/CRNN-%s-Mask-Part' % firstAttentionName
    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    classifier = CRNN_Mask(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
                           firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope)
    # classifier.Valid()

    for episode in range(100):
        print('\nTrain Episode %d Total Loss = %f' % (
            episode, classifier.Train_Part(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test_Part(logName=savepath + '-TestResult/%04d.csv' % episode, testData=trainData,
                             testLabel=trainLabel, testSeq=trainSeq)
