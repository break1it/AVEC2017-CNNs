import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.HACN import HACN
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer
from Model.AttentionMechanism.CNN_LocalAttention import CNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_StandardAttention import RNN_StandardAttentionInitializer
from Model.AttentionMechanism.RNN_LocalAttention import RNN_LocalAttentionInitializer
from Model.AttentionMechanism.RNN_MonotonicAttention import RNN_MonotonicAttentionInitializer

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = \
        Loader_CNN(partName='CNN-10')

    firstAttention = CNN_StandardAttentionInitializer
    firstAttentionScope = None
    firstAttentionName = 'CSA'
    secondAttention = RNN_StandardAttentionInitializer
    secondAttentionScope = None
    secondAttentionName = 'RSA'

    classifier = HACN(
        trainData=numpy.concatenate([trainData, developData], axis=0),
        trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0),
        firstAttention=firstAttention, firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope,
        secondAttention=secondAttention, secondAttentionName=secondAttentionName,
        secondAttentionScope=secondAttentionScope, middleLayer='None')
    classifier.Valid()
