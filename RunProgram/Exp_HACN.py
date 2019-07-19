import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.HACN import HACN
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer
from Model.AttentionMechanism.CNN_LocalAttention import CNN_LocalAttentionInitializer

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = \
        Loader_CNN(partName='CNN-10')

    firstAttention = CNN_StandardAttentionInitializer
    firstAttentionScope = 'CSA'
    firstAttentionName = None
    secondAttention = None
    secondAttentionScope = None
    secondAttentionName = None

    classifier = HACN(
        trainData=numpy.concatenate([trainData, developData], axis=0),
        trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0),
        firstAttention=firstAttention, firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope,
        secondAttention=secondAttention, secondAttentionName=secondAttentionName,
        secondAttentionScope=secondAttentionScope)
    classifier.Valid()
