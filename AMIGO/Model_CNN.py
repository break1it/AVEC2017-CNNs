import numpy
import random
import tensorflow


def Shuffle_Double(a, b):
    if len(a) != len(b):
        raise RuntimeError("Input Don't Have Same Len.")

    index = numpy.arange(0, len(a))
    random.shuffle(index)
    newA, newB = [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
    return newA, newB


class NeuralNetwork_Base:
    def __init__(self, trainData, trainLabel, batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        '''
        :param trainData:       This is the data used for Train.
        :param trainLabel:      This is the label of the train data.
        :param batchSize:       This number indicates how many samples used for one batch.
        :param learningRate:    This is the learning rate of Neural Network.
        :param startFlag:       This is the flag which decide to start a Neural Network or load parameters from files.
        :param graphRevealFlag: This is the flag which decide whether this Neural Network will generate a graph.
        :param graphPath:       if the graphRevealFlag is True, save the figure to this position.
        :param occupyRate:      Due to the reason that the free memory of GPU are sometimes not enough,
                                I use this flag to setting whether to automatic allocate the memory of GPU or
                                designate rate of GPU memory.

                                In absence, it is generally setting at -1 which mean that the program automatic
                                allocate the memory. In other wise, it is designate the rate of GPU memory occupation.
                                0 < occupyRate < 1
        '''
        self.data = trainData
        self.label = trainLabel
        self.batchSize = batchSize

        # Data Record Completed

        if occupyRate <= 0 or occupyRate >= 1:
            config = tensorflow.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tensorflow.ConfigProto(
                gpu_options=tensorflow.GPUOptions(per_process_gpu_memory_fraction=occupyRate))
        self.session = tensorflow.Session(config=config)

        # GPU Occupation Setting

        self.parameters = {}
        self.BuildNetwork(learningRate=learningRate)

        self.information = 'This is the base class of all other classes.' \
                           '\nIt"s high probability wrong if you see this information in the log Files.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

        if graphRevealFlag:
            tensorflow.summary.FileWriter(graphPath, self.session.graph)

        if startFlag:
            self.session.run(tensorflow.global_variables_initializer())

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None], name='labelInput')
        self.keepProbability = tensorflow.placeholder(dtype=tensorflow.float32, name='keepProbability')

        self.train = tensorflow.Variable(0)
        print('This is not Used.\n'
              'If you see this Information.\n'
              'That means there exists some problems.')

    def Train(self, logName):
        pass

    def Save(self, savepath):
        saver = tensorflow.train.Saver()
        saver.save(self.session, savepath)

    def Load(self, loadpath):
        saver = tensorflow.train.Saver()
        saver.restore(self.session, loadpath)

    def SaveGraph(self, graphPath):
        tensorflow.summary.FileWriter(graphPath, self.session.graph)


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
