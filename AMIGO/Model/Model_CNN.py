import numpy
import tensorflow
from AMIGO.Tools import Shuffle_Double, NeuralNetwork_Base


class CNN_SingleTask(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        super(CNN_SingleTask, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 20, 65], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2], name='labelInput')

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', name='Layer1st_Conv')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=2, padding='SAME', name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', name='Layer2nd_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=2, padding='SAME', name='Layer2nd_MaxPooling')
        self.parameters['Layer2nd_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Layer2nd_MaxPooling'], shape=[-1, 5 * 17 * 16], name='Layer2nd_Reshape')

        self.parameters['Layer3rd_FC'] = tensorflow.layers.dense(
            inputs=self.parameters['Layer2nd_Reshape'], units=128, activation=tensorflow.nn.relu, name='Layer3rd_FC')
        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['Layer3rd_FC'], units=2, activation=None, name='Predict')

        self.parameters['Loss'] = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=self.labelInput, logits=self.parameters['Predict'], weights=10)
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train_Inbalance(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

        startPosition = 0
        totalLoss = 0.0
        with open(logName, 'w') as file:
            while startPosition < numpy.shape(trainData)[0]:
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                               self.labelInput: trainLabel[startPosition:startPosition + self.batchSize]})
                print('\rTrain %d/%d Loss = %f' % (startPosition, len(trainData), loss), end='')
                totalLoss += loss
                startPosition += self.batchSize
                file.write(str(loss) + '\n')
        return totalLoss

    def Test(self, logName, testData, testLabel):
        with open(logName, 'w') as file:
            startPosition = 0
            while startPosition < len(testData):
                batchData, batchLabel = testData[startPosition:startPosition + self.batchSize], \
                                        testLabel[startPosition:startPosition + self.batchSize]
                result = self.session.run(fetches=self.parameters['Predict'], feed_dict={self.dataInput: batchData})

                for index in range(len(result)):
                    file.write(str(numpy.argmax(batchLabel[index])) + ',' + str(numpy.argmax(result[index])) + '\n')
                startPosition += self.batchSize

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Predict'],
                                  feed_dict={self.dataInput: self.data[0:self.batchSize]})
        print(numpy.shape(result))
