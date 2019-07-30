import os
import numpy
import tensorflow
from Auxiliary.Loader import Loader_CNN_Flexible
from Model.CRNN_Mask import CRNN_Mask
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer_Mask
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer_Mask
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer_Mask
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer_Mask

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_CNN_Flexible(partName='CNN-10', maxSentence=10, maxSpeech=-1)

    # print(trainSeq[0])

    firstAttention = CNN_StandardAttentionInitializer_Mask
    firstAttentionScope = None
    firstAttentionName = 'CSA'

    secondAttention = RNN_StandardAttentionInitializer_Mask
    secondAttentionScope = None
    secondAttentionName = 'RSA_MASK'

    loadpath = 'D:/PythonProjects_Data/AVEC2017-Experiment-Mask/BLSTM-Changes-CRNN-%s-%s-Part' % (
        firstAttentionName, secondAttentionName)

    classifier = CRNN_Mask(
        trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
        firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, secondAttention=secondAttention,
        secondAttentionName=secondAttentionName, secondAttentionScope=secondAttentionScope, startFlag=False)
    classifier.Load(loadpath=loadpath + '/Network-%04d' % 99)
    classifier.Valid()
