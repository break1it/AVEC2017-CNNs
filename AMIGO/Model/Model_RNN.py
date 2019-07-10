from AMIGO.Tools import NeuralNetwork_Base, Shuffle_Double
import tensorflow
from tensorflow.contrib import rnn
import numpy


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


class BLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, batchSize=32, hiddenNodules=128, rnnLayers=2, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.hiddenNodules, self.rnnLayers = hiddenNodules, rnnLayers
        self.positiveData, self.positiveLabel, self.negativeData, self.negativeLabel = [], [], [], []

        for index in range(numpy.shape(trainData)[0]):
            if numpy.argmax(trainLabel[index]) == 0:
                self.negativeData.append(trainData[index])
                self.negativeLabel.append(trainLabel[index])
            else:
                self.positiveData.append(trainData[index])
                self.positiveLabel.append(trainLabel[index])

        super(BLSTM, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 20, 65], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2], name='labelInput')

        self.parameters['Cell_FW'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['Cell_BW'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

        self.parameters['BLSTM_Output'], self.parameters['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['Cell_FW'], cell_bw=self.parameters['Cell_BW'],
                inputs=self.dataInput, dtype=tensorflow.float32)
        self.parameters['AttentionList'] = StandardAttentionInitializer(
            dataInput=self.parameters['BLSTM_Output'], scopeName='StandardAttention',
            hiddenNoduleNumber=2 * self.hiddenNodules, blstmFlag=True)
        self.parameters['AttentionResult'] = self.parameters['AttentionList']['FinalResult']

        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResult'], units=2, activation=None, name='Predict')
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

    def Train_Inbalance(self, logName):
        trainData, trainLabel = Shuffle_Double(self.data, self.label)

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

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: self.data[0:self.batchSize],
                                             self.labelInput: self.label[0:self.batchSize]})
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
                        fetches=[self.parameters['AttentionResult'], self.parameters['Predict']],
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
