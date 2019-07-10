import numpy
import tensorflow
from Auxiliary.Shuffle import Shuffle_Double
from Model.Base import NeuralNetwork_Base
from Model.AttentionMechanism.CNN_LocalAttention import CNN_LocalAttentionInitializer


class HACN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, attention, attentionName, attentionScope, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.attention, self.attentionName, self.attentionScope = attention, attentionName, attentionScope
        super(HACN, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=None, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[128, 1000, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1], name='labelInput')

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer2nd_MaxPooling')

        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameters['Layer3rd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer3rd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer3rd_MaxPooling')

        ####################################################################################

        self.parameters['FirstAttentionList'] = CNN_LocalAttentionInitializer(
            inputData=self.parameters['Layer3rd_MaxPooling'], scopeName='LocalAttention', hiddenNoduleNumbers=16)
        self.parameters['FinalAttentionResult'] = self.parameters['FirstAttentionList']['FinalResult']

        self.parameters['MediaFC'] = tensorflow.layers.dense(
            inputs=self.parameters['FinalAttentionResult'], units=128, activation=tensorflow.nn.relu, name='MediaFC')

        self.parameters['SecondAttentionList'] = self.attention(
            dataInput=self.parameters['MediaFC'][tensorflow.newaxis, :, :], scopeName=self.attentionName,
            hiddenNoduleNumber=128, attentionScope=self.attentionScope, blstmFlag=False)
        self.parameters['SecondAttentionResult'] = self.parameters['SecondAttentionList']['FinalResult']

        #####################################################################################

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['SecondAttentionResult'], units=1, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.absolute_difference(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

        totalLoss = 0.0
        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: trainData[index],
                               self.labelInput: numpy.reshape(trainLabel[index], [-1, 1])})
                file.write(str(loss) + '\n')
                totalLoss += loss
                print('\rTraining %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')
        return totalLoss

    def Test(self, logName, testData, testLabel):
        with open(logName, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                predict = self.session.run(fetches=self.parameters['Predict'],
                                           feed_dict={self.dataInput: testData[index]})
                file.write(str(testLabel[index]) + ',' + str(predict[0][0]) + '\n')

                print('\rTesting %d/%d' % (index, numpy.shape(testData)[0]), end='')

    def Valid(self):
        result = self.session.run(
            fetches=self.parameters['Loss'],
            feed_dict={self.dataInput: self.data[0], self.labelInput: numpy.reshape(self.label[0], [-1, 1])})
        print(numpy.shape(result))
        # print(result)
