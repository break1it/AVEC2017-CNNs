import os
import numpy
import tensorflow
from Auxiliary.Loader import Loader_CNN_Flexible
from Model.CRNN_Mask import CRNN_Mask
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer_Mask

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_CNN_Flexible(partName='CNN-10', maxSentence=10)

    firstAttention = CNN_StandardAttentionInitializer_Mask
    firstAttentionScope = None
    firstAttentionName = 'CSA'

    loadpath = 'D:/PythonProjects_Data/Experiment/CRNN-%s-Mask-Part/Network-%04d' % (firstAttentionName, 20)

    classifier = CRNN_Mask(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, startFlag=False)

    classifier.Load(loadpath=loadpath)
    classifier.Valid()
