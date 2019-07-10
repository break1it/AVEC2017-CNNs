import tensorflow


def RNN_LocalAttentionInitializer(dataInput, scopeName, hiddenNoduleNumber, attentionScope=None, blstmFlag=True):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        if blstmFlag:
            networkParameter['DataInput'] = tensorflow.concat([dataInput[0], dataInput[1]], axis=2, name='DataInput')
        else:
            networkParameter['DataInput'] = dataInput

        networkParameter['BatchSize'], networkParameter['TimeStep'], networkParameter[
            'HiddenNoduleNumber'] = tensorflow.unstack(tensorflow.shape(networkParameter['DataInput'], name='Shape'))
        networkParameter['DataSupplement'] = tensorflow.concat([networkParameter['DataInput'], tensorflow.zeros(
            shape=[networkParameter['BatchSize'], attentionScope * 2 + 1, networkParameter['HiddenNoduleNumber']])],
                                                               axis=1, name='DataSupplement')

        ############################################################################

        networkParameter['DataPile_0'] = tensorflow.concat(
            [networkParameter['DataSupplement'][:, 0:-2 * attentionScope - 1, :],
             networkParameter['DataSupplement'][:, 1:-2 * attentionScope, :]], axis=2, name='DataPile_0')
        for times in range(1, 2 * attentionScope):
            networkParameter['DataPile_%d' % times] = tensorflow.concat(
                [networkParameter['DataPile_%d' % (times - 1)],
                 networkParameter['DataSupplement'][:, times + 1:-2 * attentionScope + times, :]], axis=2,
                name='DataPile_%d' % times)
        networkParameter['DataPileFinal'] = networkParameter['DataPile_%d' % (2 * attentionScope - 1)]

        ############################################################################

        networkParameter['DataReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataPileFinal'],
            shape=[networkParameter['BatchSize'] * networkParameter['TimeStep'],
                   (2 * attentionScope + 1) * networkParameter['HiddenNoduleNumber']], name='DataReshape')
        networkParameter['DataReshape'].set_shape([None, (2 * attentionScope + 1) * hiddenNoduleNumber])
        networkParameter['DataWeight'] = tensorflow.layers.dense(inputs=networkParameter['DataReshape'], units=1,
                                                                 activation=tensorflow.nn.tanh,
                                                                 name='DataWeight' + scopeName)
        networkParameter['DataWeightReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataWeight'], shape=[networkParameter['BatchSize'], networkParameter['TimeStep']],
            name='DataWeightReshape')
        networkParameter['AttentionFinal'] = tensorflow.nn.softmax(logits=networkParameter['DataWeightReshape'],
                                                                   name='DataWeightSoftmax')

        networkParameter['DataWeightPile'] = tensorflow.tile(
            input=networkParameter['AttentionFinal'][:, :, tensorflow.newaxis],
            multiples=[1, 1, networkParameter['HiddenNoduleNumber']], name='DataWeightPile')
        networkParameter['FinalResultTile'] = tensorflow.multiply(
            x=networkParameter['DataInput'], y=networkParameter['DataWeightPile'], name='FinalResultTile')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(input_tensor=networkParameter['FinalResultTile'],
                                                                axis=1, name='FinalResult')

    return networkParameter

