import os
import numpy
import tensorflow
from Auxiliary.Loader import Loader_CNN_Flexible
from Model.CRNN_Mask import CRNN_Mask
from Model.AttentionMechanism.CNN_StandardAttention import CNN_StandardAttentionInitializer_Mask
import matplotlib.pylab as plt

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, developData, developLabel, developSeq, testData, testLabel, testSeq = \
        Loader_CNN_Flexible(partName='CNN-10', maxSentence=10)
    print(numpy.shape(trainData))
    data = numpy.transpose(trainData[0][0][0:115], [1, 0])
    plt.imshow(data)
    plt.axis('off')
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.savefig('output.png')
    plt.show()

    # firstAttention = CNN_StandardAttentionInitializer_Mask
    # firstAttentionScope = None
    # firstAttentionName = 'CSA'
    #
    # loadpath = 'D:/PythonProjects_Data/Experiment/CRNN-%s-Mask-Part/Network-%04d' % (firstAttentionName, 20)
    #
    # classifier = CRNN_Mask(
    #     trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, firstAttention=firstAttention,
    #     firstAttentionName=firstAttentionName, firstAttentionScope=firstAttentionScope, startFlag=False)
    #
    # classifier.Load(loadpath=loadpath)
    # classifier.Valid()
