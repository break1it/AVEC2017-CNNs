from AMIGO.Tools import NeuralNetwork_Base, Shuffle_Double
import tensorflow
from tensorflow.contrib import rnn


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

    def Train(self, logName):
        pass

    def Valid(self):
        pass
