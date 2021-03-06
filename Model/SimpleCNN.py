import numpy
import tensorflow
from Model.Base import NeuralNetwork_Base
from Auxiliary.Shuffle import Shuffle_Double


class SimpleCNN_3Layers(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, attention, attentionScope, attentionName, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.attention, self.attentionScope, self.attentioName = attention, attentionScope, attentionName
        super(SimpleCNN_3Layers, self).__init__(
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

        self.attentionList = self.attention(
            inputData=self.parameters['Layer3rd_MaxPooling'], attentionScope=self.attentionScope,
            scopeName=self.attentioName)
        self.parameters['AttentionResult'] = self.attentionList['FinalResult']
        self.parameters['AttentionResultReshape'] = tensorflow.reshape(
            tensor=self.parameters['AttentionResult'], shape=[-1, 16 * 128], name='AttentionResultReshape')

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResultReshape'], units=1, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
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
                predict = self.session.run(
                    fetches=self.parameters['Predict'],
                    feed_dict={self.dataInput: testData[index]})

                file.write(str(predict[0][0]) + ',' + str(testLabel[index]) + '\n')
                print('\rTesting %d/%d' % (index, numpy.shape(testData)[0]), end='')

    def Valid(self):
        result = self.session.run(fetches=self.parameters['AttentionResult'],
                                  feed_dict={self.dataInput: self.data[0],
                                             self.labelInput: numpy.reshape(self.label[0], [-1, 1])})
        print(numpy.shape(result))
        # print(result)


class SimpleCNN_2Layers(SimpleCNN_3Layers):
    def __init__(self, trainData, trainLabel, attention, attentionScope, attentionName, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        super(SimpleCNN_2Layers, self).__init__(
            trainData=trainData, trainLabel=trainLabel, attention=attention, attentionScope=attentionScope,
            attentionName=attentionName, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

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

        self.attentionList = self.attention(
            inputData=self.parameters['Layer2nd_MaxPooling'], attentionScope=self.attentionScope,
            scopeName=self.attentioName)
        self.parameters['AttentionResult'] = self.attentionList['FinalResult']
        self.parameters['AttentionResultReshape'] = tensorflow.reshape(
            tensor=self.parameters['AttentionResult'], shape=[-1, 16 * 128], name='AttentionResultReshape')

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResultReshape'], units=1, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: self.data[0],
                                             self.labelInput: numpy.reshape(self.label[0], [-1, 1])})
        print(numpy.shape(result))
        # print(result)


class SimpleCNN_2_5Layers(SimpleCNN_3Layers):
    def __init__(self, trainData, trainLabel, attention, attentionScope, attentionName, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        super(SimpleCNN_2_5Layers, self).__init__(
            trainData=trainData, trainLabel=trainLabel, attention=attention, attentionScope=attentionScope,
            attentionName=attentionName, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

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

        self.attentionList = self.attention(
            inputData=self.parameters['Layer3rd_Conv'], attentionScope=self.attentionScope, scopeName=self.attentioName)
        self.parameters['AttentionResult'] = self.attentionList['FinalResult']
        self.parameters['AttentionResultReshape'] = tensorflow.reshape(
            tensor=self.parameters['AttentionResult'], shape=[-1, 16 * 128], name='AttentionResultReshape')

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResultReshape'], units=1, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.huber_loss(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: self.data[0],
                                             self.labelInput: numpy.reshape(self.label[0], [-1, 1])})
        print(numpy.shape(result))
