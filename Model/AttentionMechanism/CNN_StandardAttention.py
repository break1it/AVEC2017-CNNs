import tensorflow


def CNN_StandardAttentionInitializer(inputData, attentionScope, hiddenNoduleNumber, scopeName='CSA'):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        networkParameter['DataInput'] = inputData
        networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], \
        networkParameter['HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Flat'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh, name='AttentionWeight_Flat')
        networkParameter['AttentionWeight_Reshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight_Flat'], shape=[networkParameter['BatchSize'], -1],
            name='AttentionWeight_Reshape')
        networkParameter['AttentionWeight_SoftMax'] = tensorflow.nn.softmax(
            networkParameter['AttentionWeight_Reshape'], name='AttentionWeight_SoftMax')
        networkParameter['AttentionWeight_Tile'] = tensorflow.tile(
            input=networkParameter['AttentionWeight_SoftMax'][:, :, tensorflow.newaxis],
            multiples=[1, 1, networkParameter['HiddenNoduleNumber']], name='AttentionWeight_Tile')

        networkParameter['DataReshape'] = tensorflow.reshape(
            tensor=networkParameter['DataInput'],
            shape=[networkParameter['BatchSize'], networkParameter['XScope'] * networkParameter['YScope'],
                   networkParameter['HiddenNoduleNumber']], name='DataReshape')

        networkParameter['Data_AddWeight_Raw'] = tensorflow.multiply(
            networkParameter['AttentionWeight_Tile'], networkParameter['DataReshape'], name='Data_AddWeight_Raw')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['Data_AddWeight_Raw'], axis=1, name='FinalResult')
        networkParameter['FinalResult'].set_shape([None, hiddenNoduleNumber])

    return networkParameter


def CNN_StandardAttentionInitializer_Mask(inputData, inputSeq, attentionScope, hiddenNoduleNumber, scopeName='CSA'):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        networkParameter['DataInput'] = inputData
        networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], \
        networkParameter['HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Flat'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh, name='AttentionWeight_Flat')
        networkParameter['AttentionWeight_Reshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight_Flat'], shape=[networkParameter['BatchSize'], -1],
            name='AttentionWeight_Reshape')
