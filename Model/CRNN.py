import numpy
import tensorflow
from tensorflow.contrib import rnn
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Double

PRETRAINED_SCOPE = {'CSA': 26, 'CLA': 26}


class CRNN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, firstAttention, firstAttentionScope, firstAttentionName, secondAttention,
                 secondAttentionName, secondAttentionScope, rnnLayers=1, hiddenNoduleNumber=128, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.firstAttention, self.firstAttentionScope, self.firstAttentionName = firstAttention, firstAttentionScope, firstAttentionName
        self.secondAttention, self.secondAttentionScope, self.secondAttentionName = secondAttention, secondAttentionScope, secondAttentionName
        self.rnnLayers, self.hiddenNoduleNumber = rnnLayers, hiddenNoduleNumber
        super(CRNN, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=None, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[128, 1000, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1], name='labelInput')

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 1], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=64, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv',
            kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 1], padding='SAME',
            name='Layer2nd_MaxPooling')

        self.parameters['FirstAttentionList'] = self.firstAttention(
            inputData=self.parameters['Layer2nd_MaxPooling'], attentionScope=self.firstAttentionScope,
            hiddenNoduleNumber=64, scopeName=self.firstAttentionName)
        self.parameters['FirstAttentionResult'] = self.parameters['FirstAttentionList']['FinalResult']

        self.parameters['FirstAttentionResult_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['FirstAttentionResult'], shape=[-1, 128 * 64], name='FirstAttentionResult_Reshape')
        self.parameters['Predict_Part'] = tensorflow.layers.dense(
            inputs=self.parameters['FirstAttentionResult_Reshape'], units=1, activation=None, name='Predict_Part')
        self.parameters['Loss_Part'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict_Part'])
        self.trainPart = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Loss_Part'])

        self.parameters['BLSTM_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_Output'], self.parameters['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['BLSTM_FW_Cell'], cell_bw=self.parameters['BLSTM_BW_Cell'],
                inputs=self.parameters['FirstAttentionResult'][tensorflow.newaxis, :, :], dtype=tensorflow.float32)

        self.parameters['SecondAttention'] = self.secondAttention(
            dataInput=self.parameters['BLSTM_Output'], scopeName=self.secondAttentionName,
            hiddenNoduleNumber=2 * self.hiddenNoduleNumber, attentionScope=self.secondAttentionScope, blstmFlag=True)
        self.parameters['SecondAttentionResult'] = self.parameters['SecondAttention']['FinalResult']

        self.parameters['Predict_Total'] = tensorflow.layers.dense(
            inputs=self.parameters['SecondAttentionResult'], units=1, activation=None, name='Predict_Total')
        self.parameters['Loss_Total'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict_Total'])
        self.trainTotal = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Loss_Total'],
            var_list=tensorflow.global_variables()[PRETRAINED_SCOPE[self.firstAttentionName]:])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['FirstAttentionList']['AttentionWeight'],
                                  feed_dict={self.dataInput: self.data[0],
                                             self.labelInput: numpy.reshape(self.label[0], [-1, 1])})
        print(numpy.shape(result))
        result = numpy.reshape(result[0], [250, 40])
        result = numpy.transpose(result, [1, 0])
        import matplotlib.pylab as plt
        plt.imshow(result)
        plt.xlabel('Frames')
        plt.ylabel('Bands')
        plt.colorbar()
        plt.show()
        # print(result)

    def Train_Total(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

        totalLoss = 0.0
        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss_Total'], self.trainTotal],
                    feed_dict={self.dataInput: trainData[index],
                               self.labelInput: numpy.reshape(trainLabel[index], [-1, 1])})
                file.write(str(loss) + '\n')
                totalLoss += loss

                print('\rTraining %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')

        return totalLoss

    def Test_Total(self, logName, testData, testLabel):
        with open(logName, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                predict = self.session.run(
                    fetches=self.parameters['Predict_Total'],
                    feed_dict={self.dataInput: testData[index]})

                file.write(str(predict[0][0]) + ',' + str(testLabel[index]) + '\n')
                print('\rTesting %d/%d' % (index, numpy.shape(testData)[0]), end='')

    def Train_Part(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

        totalLoss = 0.0
        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss_Part'], self.trainPart],
                    feed_dict={self.dataInput: trainData[index],
                               self.labelInput: numpy.reshape(trainLabel[index], [-1, 1])})
                file.write(str(loss) + '\n')
                totalLoss += loss

                print('\rTraining %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')

        return totalLoss

    def Test_Part(self, logName, testData, testLabel):
        with open(logName, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                predict = self.session.run(
                    fetches=self.parameters['Predict_Part'],
                    feed_dict={self.dataInput: testData[index]})

                file.write(str(predict[0][0]) + ',' + str(testLabel[index]) + '\n')
                print('\rTesting %d/%d' % (index, numpy.shape(testData)[0]), end='')

    def LoadPart(self, loadpath):
        saver = tensorflow.train.Saver(
            var_list=tensorflow.global_variables()[0:PRETRAINED_SCOPE[self.firstAttentionName]])
        saver.restore(self.session, loadpath)
