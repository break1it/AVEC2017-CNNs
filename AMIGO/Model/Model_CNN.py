import numpy
import tensorflow
from AMIGO.Tools import Shuffle_Double, NeuralNetwork_Base


class CNN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, layers=3, batchSize=32, learningRate=1E-3, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.layers = layers
        self.positiveData, self.positiveLabel, self.negativeData, self.negativeLabel = [], [], [], []

        for index in range(numpy.shape(trainData)[0]):
            if numpy.argmax(trainLabel[index]) == 0:
                self.negativeData.append(trainData[index])
                self.negativeLabel.append(trainLabel[index])
            else:
                self.positiveData.append(trainData[index])
                self.positiveLabel.append(trainLabel[index])

        super(CNN, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 20 * 65], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2], name='labelInput')

        self.parameters['Layer1st_Conv1d'] = tensorflow.layers.conv1d(
            inputs=self.dataInput[:, :, tensorflow.newaxis], filters=16, kernel_size=16, strides=1, padding='SAME',
            name='Layer1st_Conv1d')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling1d(
            inputs=self.parameters['Layer1st_Conv1d'], pool_size=3, strides=2, padding='SAME',
            name='Layer1st_MaxPooling')

        if self.layers > 1:
            self.parameters['Layer2nd_Conv1d'] = tensorflow.layers.conv1d(
                inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=16, strides=1, padding='SAME',
                name='Layer2nd_Conv1d')
            self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling1d(
                inputs=self.parameters['Layer2nd_Conv1d'], pool_size=3, strides=2, padding='SAME',
                name='Layer2nd_MaxPooling')

            if self.layers > 2:
                self.parameters['Layer3rd_Conv1d'] = tensorflow.layers.conv1d(
                    inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=16, strides=1,
                    padding='SAME', name='Layer3rd_Conv1d')
                self.parameters['Layer3rd_MaxPooling'] = tensorflow.layers.max_pooling1d(
                    inputs=self.parameters['Layer3rd_Conv1d'], pool_size=3, strides=2, padding='SAME',
                    name='Layer3rd_MaxPooling')

        if self.layers == 1:
            self.parameters['Reshape'] = tensorflow.reshape(
                tensor=self.parameters['Layer1st_MaxPooling'], shape=[-1, 650 * 16], name='Reshape')
        if self.layers == 2:
            self.parameters['Reshape'] = tensorflow.reshape(
                tensor=self.parameters['Layer2nd_MaxPooling'], shape=[-1, 325 * 16], name='Reshape')
        if self.layers == 3:
            self.parameters['Reshape'] = tensorflow.reshape(
                tensor=self.parameters['Layer3rd_MaxPooling'], shape=[-1, 163 * 16], name='Reshape')

        self.parameters['BottleNeck'] = tensorflow.layers.dense(
            inputs=self.parameters['Reshape'], units=128, activation=tensorflow.nn.relu, name='BottleNeck')

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['BottleNeck'], units=2, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=self.labelInput, logits=self.parameters['Predict'], weights=10)

        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self, logName):
        negativeData, negativeLabel = Shuffle_Double(self.negativeData, self.negativeLabel)

        trainData, trainLabel = \
            numpy.concatenate([self.positiveData, negativeData[0:numpy.shape(self.positiveData)[0]]], axis=0), \
            numpy.concatenate([self.positiveLabel, negativeLabel[0:numpy.shape(self.positiveLabel)[0]]], axis=0)

        trainData, trainLabel = Shuffle_Double(trainData, trainLabel)

        with open(logName, 'w') as file:
            startPosition = 0
            totalLoss = 0.0
            while startPosition < numpy.shape(trainData)[0]:
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train], feed_dict={
                    self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                    self.labelInput: trainLabel[startPosition:startPosition + self.batchSize]})
                startPosition += self.batchSize

                print('\rTrain %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                file.write(str(loss) + '\n')
                totalLoss += loss
        return totalLoss

    def Test(self, logName, testData, testLabel):
        with open(logName, 'w') as file:
            startPosition = 0

            while startPosition < numpy.shape(testData)[0]:
                batchData, batchLabel = testData[startPosition:startPosition + self.batchSize], \
                                        testLabel[startPosition:startPosition + self.batchSize]
                predict = self.session.run(fetches=self.parameters['Predict'], feed_dict={self.dataInput: batchData})

                for index in range(len(predict)):
                    file.write('%f,%f\n' % (numpy.argmax(batchLabel[index]), numpy.argmax(predict[index])))
                startPosition += self.batchSize

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: self.data[0:self.batchSize],
                                             self.labelInput: self.label[0:self.batchSize]})
        print(result)
        print(numpy.shape(result))
