import numpy
import tensorflow
from tensorflow.contrib import rnn
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Triple


class CRNN_Mask(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, firstAttention, firstAttentionScope, firstAttentionName,
                 secondAttention=None, secondAttentionName=None, secondAttentionScope=None, rnnLayers=1,
                 hiddenNoduleNumber=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        self.seq = trainSeq
        self.firstAttention, self.firstAttentionScope, self.firstAttentionName = firstAttention, firstAttentionScope, firstAttentionName
        self.secondAttention, self.secondAttentionScope, self.secondAttentionName = secondAttention, secondAttentionScope, secondAttentionName
        self.rnnLayers, self.hiddenNoduleNumber = rnnLayers, hiddenNoduleNumber
        super(CRNN_Mask, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=None, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1000, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1], name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]

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

        ###################################################################################

        self.parameters['FirstAttentionList'] = self.firstAttention(
            inputData=self.parameters['Layer2nd_MaxPooling'], inputSeq=self.seqInput,
            attentionScope=self.firstAttentionScope, hiddenNoduleNumber=64, scopeName='CSA')
        self.parameters['FirstAttentionResult'] = self.parameters['FirstAttentionList']['FinalResult']

        self.parameters['FirstAttentionResult_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['FirstAttentionResult'], shape=[-1, 128 * 64],
            name='FirstAttentionResult_Reshape')
        self.parameters['Predict_Part'] = tensorflow.layers.dense(
            inputs=self.parameters['FirstAttentionResult_Reshape'], units=1, activation=None, name='Predict_Part')
        self.parameters['Loss_Part'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict_Part'])
        self.trainPart = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Loss_Part'])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['FirstAttentionList']['AttentionWeight_SoftMax'],
                                  feed_dict={self.dataInput: self.data[0],
                                             self.labelInput: numpy.reshape(self.label[0], [-1, 1]),
                                             self.seqInput: self.seq[0]})[0]
        result = numpy.reshape(result, [250, 40])
        result = numpy.transpose(result, [1, 0])
        import matplotlib.pylab as plt
        plt.imshow(result[:, 0:50])
        plt.colorbar()
        plt.show()
        # print(result)

    def Train_Part(self, logName):
        trainData, trainLabel, trainSeq = Shuffle_Triple(self.data, self.label, self.seq)

        totalLoss = 0.0
        with open(logName, 'w') as file:
            for index in range(numpy.shape(trainData)[0]):
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss_Part'], self.trainPart],
                    feed_dict={self.dataInput: trainData[index], self.seqInput: trainSeq[index],
                               self.labelInput: numpy.reshape(trainLabel[index], [-1, 1])})
                file.write(str(loss) + '\n')
                totalLoss += loss

                print('\rTraining %d/%d Loss = %f' % (index, numpy.shape(trainData)[0], loss), end='')

        return totalLoss

    def Test_Part(self, logName, testData, testLabel, testSeq):
        with open(logName, 'w') as file:
            for index in range(numpy.shape(testData)[0]):
                predict = self.session.run(
                    fetches=self.parameters['Predict_Part'],
                    feed_dict={self.dataInput: testData[index], self.seqInput: testSeq[index]})

                file.write(str(predict[0][0]) + ',' + str(testLabel[index]) + '\n')
                print('\rTesting %d/%d' % (index, numpy.shape(testData)[0]), end='')
