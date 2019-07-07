import numpy
import random
import tensorflow
from tensorflow.contrib import rnn


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


def StandardAttentionInitializer(dataInput, scopeName, hiddenNoduleNumber, blstmFlag=True):
    '''
    This is the realization of Standard Attention in Encoder-Decoder Network Structure.
    :param dataInput:                   This is the result of BLSTM or LSTM. This Attention Mechanism will
                                                    uses this output to calculate the weights of every frames.
    :param scopeName:                   In order to avoid the same name in the whole network, this Attention
                                                    Mechanism uses this scope name to identify it.
    :param hiddenNoduleNumber:      Due to the shape size should be resigned by a variant rather than a TensorFlow
                                                     Type value, this Attention Mechanism will uses this variable to reshape the
                                                     total result of dataInput.
    :param blstmFlag:                     This is the flag to Identify dataInput stem from BLSTM Structure or
                                                     LSTM Structure.
                                                     If it is the result of BLSTM Network, this Attention Mechanism will concat
                                                     the forward result with the backward result.
    :return:
    '''
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['DataReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataInput'],
            shape=[networkParameter['BatchSize'] * networkParameter['TimeStep'],
                   networkParameter['HiddenNoduleNumber']],
            name='Reshape')
        networkParameter['DataReshape'].set_shape([None, hiddenNoduleNumber])

        networkParameter['AttentionWeight'] = tensorflow.layers.dense(
            inputs=networkParameter['DataReshape'], units=1, activation=tensorflow.nn.tanh,
            name='Weight_%s' % scopeName)

        networkParameter['AttentionReshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight'],
            shape=[networkParameter['BatchSize'], networkParameter['TimeStep']],
            name='WeightReshape')
        networkParameter['AttentionFinal'] = tensorflow.nn.softmax(logits=networkParameter['AttentionReshape'],
                                                                   name='AttentionFinal')

        networkParameter['AttentionSupplement'] = tensorflow.tile(
            input=networkParameter['AttentionFinal'][:, :, tensorflow.newaxis],
            multiples=[1, 1, hiddenNoduleNumber],
            name='AttentionSupplement')
        networkParameter['FinalResult_Media'] = tensorflow.multiply(x=networkParameter['DataInput'],
                                                                    y=networkParameter['AttentionSupplement'],
                                                                    name='FinalResult_Media')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(input_tensor=networkParameter['FinalResult_Media'],
                                                                axis=1, name='FinalResult')

    return networkParameter


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


class CRNN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, batchSize=32, hiddenNoduleNumbers=128, rnnLayers=2, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.hiddenNodules, self.rnnLayers = hiddenNoduleNumbers, rnnLayers

        # print(numpy.shape(trainLabel))
        label = []
        for index in range(numpy.shape(trainLabel)[0]):
            if trainLabel[index] == 0:
                label.append([1, 0])
            else:
                label.append([0, 1])
        # print(numpy.shape(label))

        super(CRNN, self).__init__(
            trainData=trainData, trainLabel=label, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

        self.positiveData, self.positiveLabel, self.negativeData, self.negativeLabel = [], [], [], []
        for index in range(numpy.shape(trainData)[0]):
            if self.label[index][0] == 0:
                self.positiveData.append(self.data[index])
                self.positiveLabel.append(self.label[index])
            else:
                self.negativeData.append(self.data[index])
                self.negativeLabel.append(self.label[index])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2560, 17], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2], name='labelInput')

        ##############################################################################################
        # Input Part Completed
        # CNN Part Started
        ##############################################################################################

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 1], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 1], padding='SAME',
            name='Layer2nd_MaxPooling')

        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameters['Layer3rd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer3rd_Conv'], pool_size=3, strides=[2, 1], padding='SAME',
            name='Layer3rd_MaxPooling')
        self.parameters['Layer3rd_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Layer3rd_MaxPooling'], shape=[-1, 320, 17 * 16], name='Layer3rd_Reshape')

        ##############################################################################################
        # CNN Part Completed
        # BLSTM Part Started
        ##############################################################################################

        self.parameters['ForwardCell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BackwardCell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['BLSTM_Output'], self.parameters['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['ForwardCell'], cell_bw=self.parameters['BackwardCell'],
                inputs=self.parameters['Layer3rd_Reshape'], dtype=tensorflow.float32)
        self.parameters['AttentionList'] = StandardAttentionInitializer(
            dataInput=self.parameters['BLSTM_Output'], scopeName='StandardAttention',
            hiddenNoduleNumber=2 * self.hiddenNodules, blstmFlag=True)

        ##############################################################################################
        # BLSTM Part Completed
        # Final Loss Part Started
        ##############################################################################################

        self.parameters['AttentionFinalResult'] = self.parameters['AttentionList']['FinalResult']
        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionFinalResult'], units=2, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=self.labelInput, logits=self.parameters['Predict'], weights=10)
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self, logName):
        trainNegativeData, trainNegativeLabel = Shuffle_Double(self.negativeData, self.negativeLabel)
        trainData, trainLabel = \
            numpy.concatenate([self.positiveData, trainNegativeData[0:numpy.shape(self.positiveData)[0]]], axis=0), \
            numpy.concatenate([self.positiveLabel, trainNegativeLabel[0:numpy.shape(self.positiveLabel)[0]]], axis=0)
        # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))

        trainData, trainLabel = Shuffle_Double(trainData, trainLabel)
        # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))

        #####################################################################################

        startPosition = 0
        totalLoss = 0.0

        with open(logName, 'w') as file:
            while startPosition < numpy.shape(trainData)[0]:
                batchData, batchLabel = trainData[startPosition:startPosition + self.batchSize], \
                                        trainLabel[startPosition:startPosition + self.batchSize]

                #     print(batchArousal[index], batchValence[index])

                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                           feed_dict={self.dataInput: batchData, self.labelInput: batchLabel})
                totalLoss += loss
                file.write(str(loss) + '\n')

                startPosition += self.batchSize
                print('\rTraining %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                # exit()
        return totalLoss

    def Valid(self):
        result = self.session.run(fetches=self.parameters['BLSTM_FinalState'],
                                  feed_dict={self.dataInput: self.data[0:self.batchSize]})
        print(result)
        print(numpy.shape(result))

    def Test(self, logName, testData, testLabel):
        startPosition = 0
        with open(logName, 'w') as file:
            while startPosition < numpy.shape(testData)[0]:
                batchData, batchLabel = testData[startPosition:startPosition + self.batchSize], \
                                        testLabel[startPosition:startPosition + self.batchSize]
                predict = self.session.run(
                    fetches=self.parameters['Predict'], feed_dict={self.dataInput: batchData})
                predict = numpy.argmax(predict, axis=1)
                for index in range(len(batchLabel)):
                    file.write(str(batchLabel[index]) + ',' + str(predict[index]) + '\n')
                startPosition += self.batchSize

    def Train_Origin(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

        startPosition = 0
        totalLoss = 0.0

        with open(logName, 'w') as file:
            while startPosition < numpy.shape(trainData)[0]:
                batchData, batchLabel = trainData[startPosition:startPosition + self.batchSize], \
                                        trainLabel[startPosition:startPosition + self.batchSize]

                batchArousal, batchValence = [], []
                for index in range(numpy.shape(batchLabel)[0]):
                    if batchLabel[index][0] == 0:
                        batchArousal.append([1, 0])
                    else:
                        batchArousal.append([0, 1])
                    if batchLabel[index][1] == 0:
                        batchValence.append([1, 0])
                    else:
                        batchValence.append([0, 1])

                # for index in range(numpy.shape(batchLabel)[0]):
                #     print(batchArousal[index], batchValence[index])

                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                           feed_dict={self.dataInput: batchData, self.arousalInput: batchArousal,
                                                      self.valenceInput: batchValence})
                totalLoss += loss
                file.write(str(loss) + '\n')

                startPosition += self.batchSize
                print('\rTraining %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                # exit()
        return totalLoss

    def Test_Origin(self, logName, testData, testLabel):
        with open(logName, 'w') as file:
            startPosition = 0
            while startPosition < numpy.shape(testData)[0]:
                print('\rTesting %d/%d' % (startPosition, len(testData)), end='')
                arousalPredict, valencePredict = self.session.run(
                    fetches=[self.parameters['ArousalPredict'], self.parameters['ValencePredict']],
                    feed_dict={self.dataInput: testData[startPosition:startPosition + self.batchSize]})

                arousalPredict = numpy.argmax(arousalPredict, axis=1)
                valencePredict = numpy.argmax(valencePredict, axis=1)

                batchLabel = testLabel[startPosition:startPosition + self.batchSize]
                for index in range(len(batchLabel)):
                    file.write('%d,%d,%d,%d\n' % (
                        batchLabel[index][0], batchLabel[index][1], arousalPredict[index], valencePredict[index]))

                startPosition += self.batchSize
