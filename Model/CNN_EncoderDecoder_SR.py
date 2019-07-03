import numpy
import tensorflow
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from Model.Base import NeuralNetwork_Base
from Model.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from Auxiliary.Shuffle import Shuffle_Double

VOCABULAR = 41


class CNN_EncoderDecoder(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, rnnLayers=2, batchSize=32, learningRate=1E-3, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.rnnLayers = rnnLayers
        super(CNN_EncoderDecoder, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1000, 40], name='dataInput')
        self.labelInputSR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, 125], name='labelInputSR')

        self.parameters['BatchSize'], _, _ = tensorflow.unstack(tensorflow.shape(self.dataInput))

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=8, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer1st_MaxPooling')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer1st_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer2nd_MaxPooling')

        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['Layer2nd_MaxPooling'], filters=16, kernel_size=[3, 3], strides=[1, 1],
            padding='SAME', activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameters['Layer3rd_MaxPooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameters['Layer3rd_Conv'], pool_size=3, strides=[2, 2], padding='SAME',
            name='Layer3rd_MaxPooling')

        self.parameters['Layer3rd_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Layer3rd_MaxPooling'], shape=[-1, 125, 80], name='Layer3rd_Reshape')
        self.parameters['Layer3rd_DecoderInput'] = tensorflow.layers.dense(
            inputs=self.parameters['Layer3rd_Reshape'], units=256, activation=tensorflow.nn.relu,
            name='Layer3rd_DecoderInput')

        ########################################################################################

        self.attentionList = StandardAttentionInitializer(
            inputData=self.parameters['Layer3rd_MaxPooling'], scopeName='SA', hiddenNoduleNumbers=16)
        self.parameters['AttentionResult'] = self.attentionList['FinalResult']
        self.parameters['AttentionResult'].set_shape([None, 16])

        self.parameters['EncoderVector'] = tensorflow.layers.dense(
            inputs=self.parameters['AttentionResult'], units=256, activation=tensorflow.nn.relu, name='EncoderVector')

        ########################################################################################

        self.parameters['Decoder_InitialState'] = []
        for index in range(self.rnnLayers):
            self.parameters['Encoder_Cell_Layer%d' % index] = rnn.LSTMStateTuple(
                c=self.parameters['EncoderVector'], h=self.parameters['EncoderVector'])
            self.parameters['Decoder_InitialState'].append(self.parameters['Encoder_Cell_Layer%d' % index])
        self.parameters['Decoder_InitialState'] = tuple(self.parameters['Decoder_InitialState'])

        self.parameters['DecoderSeqLen'] = tensorflow.tile(
            input=tensorflow.constant(125, dtype=tensorflow.int32, shape=[1]), multiples=[self.parameters['BatchSize']],
            name='DecoderSeqLen')

        self.parameters['Decoder_Helper'] = seq2seq.TrainingHelper(
            inputs=self.parameters['Layer3rd_DecoderInput'], sequence_length=self.parameters['DecoderSeqLen'],
            name='Decoder_Helper')

        with tensorflow.variable_scope('Decoder'):
            self.parameters['Decoder_FC'] = Dense(VOCABULAR)

            self.parameters['Decoder_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=256) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['Decoder'] = seq2seq.BasicDecoder(cell=self.parameters['Decoder_Cell'],
                                                              helper=self.parameters['Decoder_Helper'],
                                                              initial_state=self.parameters['Decoder_InitialState'],
                                                              output_layer=self.parameters['Decoder_FC'])

            self.parameters['Decoder_Logits'], self.parameters['Decoder_FinalState'], self.parameters[
                'Decoder_FinalSeq'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder'])

        with tensorflow.name_scope('Loss'):
            self.parameters['TargetsReshape'] = tensorflow.reshape(tensor=self.labelInputSR, shape=[-1],
                                                                   name='TargetsReshape')
            self.parameters['Decoder_Reshape'] = tensorflow.reshape(self.parameters['Decoder_Logits'].rnn_output,
                                                                    [-1, VOCABULAR], name='Decoder_Reshape')
            self.parameters['Cost'] = tensorflow.losses.sparse_softmax_cross_entropy(
                labels=self.parameters['TargetsReshape'], logits=self.parameters['Decoder_Reshape'])

            self.trainEncoderDecoder = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
                self.parameters['Cost'])

    def Train(self, logName):
        with open(logName, 'w') as file:
            trainData, trainLabel = Shuffle_Double(self.data, self.label)

            totalLoss = 0.0
            startPosition = 0
            while startPosition < numpy.shape(trainData)[0]:
                batchData, batchLabel = self.data[startPosition:startPosition + self.batchSize], []
                for index in range(min(self.batchSize, numpy.shape(trainData)[0] - startPosition)):
                    if len(trainLabel[startPosition + index]) < 125:
                        batchLabel.append(numpy.concatenate(
                            [trainLabel[startPosition + index],
                             numpy.zeros(125 - len(trainLabel[startPosition + index]))]))
                    else:
                        batchLabel.append(trainLabel[startPosition + index][0:125])
                # print(numpy.shape(batchData), numpy.shape(batchLabel))

                loss, _ = self.session.run(fetches=[self.parameters['Cost'], self.trainEncoderDecoder],
                                           feed_dict={self.dataInput: batchData, self.labelInputSR: batchLabel})
                print('\rTraining %d/%d Loss = %f' % (startPosition, numpy.shape(trainData)[0], loss), end='')
                startPosition += self.batchSize
                totalLoss += loss
                file.write(str(loss) + '\n')
        return totalLoss

    def Valid(self):
        batchData, batchLabel = self.data[0:self.batchSize], []

        for index in range(self.batchSize):
            batchLabel.append(numpy.concatenate([self.label[index], numpy.zeros(125 - len(self.label[index]))]))

        result = self.session.run(
            fetches=self.parameters['Cost'],
            feed_dict={self.dataInput: batchData, self.labelInputSR: batchLabel})
        print(result)
        # print(numpy.shape(result[0]))
