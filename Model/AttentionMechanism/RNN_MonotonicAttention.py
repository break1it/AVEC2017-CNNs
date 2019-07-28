import tensorflow
from tensorflow.contrib import signal
import numpy


def RNN_MonotonicAttentionInitializer(dataInput, scopeName, hiddenNoduleNumber, attentionScope=None, blstmFlag=True):
    def MovingSum(tensor, backward, forward, namescope):
        with tensorflow.name_scope(namescope):
            networkParameter['%s_Pad' % namescope] = tensorflow.pad(tensor, [[0, 0], [backward, forward]],
                                                                    name='%s_Pad' % namescope)
            networkParameter['%s_Expand' % namescope] = tensorflow.expand_dims(
                input=networkParameter['%s_Pad' % namescope], axis=-1, name='%s_Expand' % namescope)
            networkParameter['%s_Filter' % namescope] = tensorflow.ones(
                shape=[backward + forward + 1, 1, 1], dtype=tensorflow.float32, name='%s_Filter' % namescope)
            networkParameter['%s_Sum' % namescope] = tensorflow.nn.conv1d(
                value=networkParameter['%s_Expand' % namescope], filters=networkParameter['%s_Filter' % namescope],
                stride=1, padding='VALID', name='%s_Sum' % namescope)
            networkParameter['%s_Result' % namescope] = networkParameter['%s_Sum' % namescope][..., 0]

    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Denominator_Raw'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.exp,
            name='%s_AttentionWeight_Denominator_Raw' % scopeName)
        networkParameter['AttentionWeight_Denominator'] = tensorflow.maximum(tensorflow.maximum(
            x=networkParameter['AttentionWeight_Denominator_Raw'], y=1E-5, name='AttentionWeight_Denominator')[..., 0],
                                                                             1E-5)

        MovingSum(tensor=networkParameter['AttentionWeight_Denominator'], backward=attentionScope - 1, forward=0,
                  namescope='Denominator')

        networkParameter['AttentionWeight_Numerator'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh,
            name='%s_AttentionWeight_Numerator' % scopeName)[..., 0]

        networkParameter['AttentionWeight_Final'] = tensorflow.divide(
            x=networkParameter['AttentionWeight_Numerator'],
            y=tensorflow.maximum(networkParameter['AttentionWeight_Denominator'], 1E-5),
            name='AttentionWeight_Final')
        MovingSum(tensor=networkParameter['AttentionWeight_Final'], backward=0, forward=attentionScope - 1,
                  namescope='Probability')

        networkParameter['Probability_Supplement'] = tensorflow.tile(
            input=networkParameter['Probability_Result'][:, :, tensorflow.newaxis],
            multiples=[1, 1, hiddenNoduleNumber], name='Probability_Supplement')
        networkParameter['FinalResult_Media'] = tensorflow.multiply(x=networkParameter['DataInput'],
                                                                    y=networkParameter['Probability_Supplement'],
                                                                    name='FinalResult_Media')
        networkParameter['FinalResult'] = tensorflow.reduce_mean(input_tensor=networkParameter['FinalResult_Media'],
                                                                 axis=1, name='FinalResult')

    return networkParameter


def RNN_MonotonicAttentionInitializer_Mask(dataInput, scopeName, seqInput, hiddenNoduleNumber, attentionScope=None,
                                           blstmFlag=True):
    def MovingSum(tensor, backward, forward, namescope):
        with tensorflow.name_scope(namescope):
            networkParameter['%s_Pad' % namescope] = tensorflow.pad(tensor, [[0, 0], [backward, forward]],
                                                                    name='%s_Pad' % namescope)
            networkParameter['%s_Expand' % namescope] = tensorflow.expand_dims(
                input=networkParameter['%s_Pad' % namescope], axis=-1, name='%s_Expand' % namescope)
            networkParameter['%s_Filter' % namescope] = tensorflow.ones(
                shape=[backward + forward + 1, 1, 1], dtype=tensorflow.float32, name='%s_Filter' % namescope)
            networkParameter['%s_Sum' % namescope] = tensorflow.nn.conv1d(
                value=networkParameter['%s_Expand' % namescope], filters=networkParameter['%s_Filter' % namescope],
                stride=1, padding='VALID', name='%s_Sum' % namescope)
            networkParameter['%s_Result' % namescope] = networkParameter['%s_Sum' % namescope][..., 0]

    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Denominator_Raw'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.exp,
            name='%s_AttentionWeight_Denominator_Raw' % scopeName)
        networkParameter['AttentionWeight_Denominator'] = tensorflow.maximum(tensorflow.maximum(
            x=networkParameter['AttentionWeight_Denominator_Raw'], y=1E-5, name='AttentionWeight_Denominator')[..., 0],
                                                                             1E-5)

        MovingSum(tensor=networkParameter['AttentionWeight_Denominator'], backward=attentionScope - 1, forward=0,
                  namescope='Denominator')

        networkParameter['AttentionWeight_Numerator'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=None,
            name='%s_AttentionWeight_Numerator' % scopeName)[..., 0]

        networkParameter['AttentionWeight_Final'] = tensorflow.divide(
            x=networkParameter['AttentionWeight_Numerator'],
            y=tensorflow.maximum(networkParameter['AttentionWeight_Denominator'], 1E-5),
            name='AttentionWeight_Final')
        MovingSum(tensor=networkParameter['AttentionWeight_Final'], backward=0, forward=attentionScope - 1,
                  namescope='Probability')

        networkParameter['Probability_Supplement'] = tensorflow.tile(
            input=networkParameter['Probability_Result'][:, :, tensorflow.newaxis],
            multiples=[1, 1, hiddenNoduleNumber], name='Probability_Supplement')
        networkParameter['FinalResult_Media'] = tensorflow.multiply(x=networkParameter['DataInput'],
                                                                    y=networkParameter['Probability_Supplement'],
                                                                    name='FinalResult_Media')
        networkParameter['FinalResult'] = tensorflow.reduce_mean(input_tensor=networkParameter['FinalResult_Media'],
                                                                 axis=1, name='FinalResult')

    return networkParameter
