from AMIGO.Tools import NeuralNetwork_Base, Shuffle_Double
import tensorflow
import numpy


class NeuralNetwork(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.numShape = numpy.shape(trainData)[1]
        self.positiveData, self.positiveLabel, self.negativeData, self.negativeLabel = [], [], [], []

        for index in range(numpy.shape(trainData)[0]):
            if numpy.argmax(trainLabel[index]) == 0:
                self.negativeData.append(trainData[index])
                self.negativeLabel.append(trainLabel[index])
            else:
                self.positiveData.append(trainData[index])
                self.positiveLabel.append(trainLabel[index])

        super(NeuralNetwork, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.numShape], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2], name='labelInput')

        self.parameters['Layer1st_FC'] = tensorflow.layers.dense(
            inputs=self.dataInput, units=1024, activation=tensorflow.nn.relu, name='Layer1st_FC')
        self.parameters['Layer2nd_FC'] = tensorflow.layers.dense(
            inputs=self.parameters['Layer1st_FC'], units=1024, activation=tensorflow.nn.relu, name='Layer2nd_FC')
        self.parameters['Layer3rd_FC'] = tensorflow.layers.dense(
            inputs=self.parameters['Layer2nd_FC'], units=1024, activation=tensorflow.nn.relu, name='Layer3rd_FC')
        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['Layer3rd_FC'], units=2, activation=None, name='Predict')

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
            startPosition, totalLoss = 0, 0.0
            while startPosition < numpy.shape(trainData)[0]:
                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                               self.labelInput: trainLabel[startPosition:startPosition + self.batchSize]})
                startPosition += self.batchSize
                file.write(str(loss) + '\n')
                print('\rTrain %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
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
        result = self.session.run(
            fetches=self.parameters['Loss'],
            feed_dict={self.dataInput: self.data[0:self.batchSize], self.labelInput: self.label[0:self.batchSize]})
        print(result)
        print(numpy.shape(result))

    def MiddleResult(self, logName, data, label):
        with open(logName + '-Label.csv', 'w') as file:
            for indexX in range(numpy.shape(label)[0]):
                for indexY in range(numpy.shape(label)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(label[indexX][indexY]))
                file.write('\n')
        with open(logName + '-FC.csv', 'w') as fileFC:
            with open(logName + '-Predict.csv', 'w') as filePredict:
                startPosition = 0

                while startPosition < numpy.shape(data)[0]:
                    batchData = data[startPosition:startPosition + self.batchSize]

                    FCResult, PredictResult = self.session.run(
                        fetches=[self.parameters['Layer3rd_FC'], self.parameters['Predict']],
                        feed_dict={self.dataInput: batchData})

                    for indexX in range(numpy.shape(FCResult)[0]):
                        for indexY in range(numpy.shape(FCResult)[1]):
                            if indexY != 0: fileFC.write(',')
                            fileFC.write(str(FCResult[indexX][indexY]))
                        fileFC.write('\n')

                    for indexX in range(numpy.shape(PredictResult)[0]):
                        for indexY in range(numpy.shape(PredictResult)[1]):
                            if indexY != 0: filePredict.write(',')
                            filePredict.write(str(PredictResult[indexX][indexY]))
                        filePredict.write('\n')

                    startPosition += self.batchSize
