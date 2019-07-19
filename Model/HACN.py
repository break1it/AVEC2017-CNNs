import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Double


class HACN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, firstAttention, firstAttentionScope, firstAttentionName, secondAttention,
                 secondAttentionName, secondAttentionScope, middleLayer='FC', hiddenNoduleNumber=128, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.firstAttention, self.firstAttentionScope, self.firstAttentionName = firstAttention, firstAttentionScope, firstAttentionName
        self.secondAttention, self.secondAttentionScope, self.secondAttentionName = secondAttention, secondAttentionScope, secondAttentionName
        self.middleLayer, self.hiddenNoduleNumber = middleLayer, hiddenNoduleNumber
        super(HACN, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=None, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[128, 1000, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1], name='labelInput')

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer2nd_MaxPooling')

        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer3rd_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer3rd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer3rd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer3rd_MaxPooling')

        self.parameters['FirstAttentionList'] = self.firstAttention(
            inputData=self.parameters['Layer3rd_MaxPooling'], attentionScope=self.firstAttentionScope,
            hiddenNoduleNumber=16, scopeName=self.firstAttentionName)
        self.parameters['FirstAttentionResult'] = self.parameters['FirstAttentionList']['FinalResult']

        if self.middleLayer == 'FC':
            self.parameters['SecondAttentionInput'] = tensorflow.layers.dense(
                inputs=self.parameters['FirstAttentionResult'], units=self.hiddenNoduleNumber,
                activation=tensorflow.nn.relu, name='SecondAttentionInput')
            nodules = self.hiddenNoduleNumber
        if self.middleLayer == 'None':
            self.parameters['SecondAttentionInput'] = self.parameters['FirstAttentionResult']
            nodules = 16

        self.parameters['SecondAttention'] = self.secondAttention(
            dataInput=self.parameters['SecondAttentionInput'][tensorflow.newaxis, :, :],
            scopeName=self.secondAttentionName, hiddenNoduleNumber=nodules, attentionScope=self.secondAttentionScope,
            blstmFlag=False)
        self.parameters['SecondAttentionResult'] = self.parameters['SecondAttention']['FinalResult']

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['SecondAttentionResult'], units=1, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: self.data[0],
                                             self.labelInput: numpy.reshape(self.label[0], [-1, 1])})
        print(numpy.shape(result))
        # print(result)
