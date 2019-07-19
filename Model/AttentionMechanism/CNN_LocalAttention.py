import tensorflow


def CNN_LocalAttentionInitializer(inputData, attentionScope, hiddenNoduleNumber, scopeName='CSA'):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        networkParameter['DataInput'] = inputData
        networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], \
        networkParameter['HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight'] = tensorflow.layers.conv2d(
            inputs=networkParameter['DataInput'], filters=1, kernel_size=attentionScope, strides=[1, 1], padding='SAME',
            name='AttentionWeight', kernel_initializer=tensorflow.random_normal_initializer(mean=0.0, stddev=0.1))
        networkParameter['AttentionWeight_Reshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight'], shape=[networkParameter['BatchSize'], -1, 1],
            name='AttentionWeight_Reshape')
        networkParameter['AttentionWeight_SoftMax'] = tensorflow.nn.softmax(
            networkParameter['AttentionWeight_Reshape'], name='AttentionWeight_SoftMax')
        networkParameter['AttentionWeight_OriginShape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight_SoftMax'],
            shape=[networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], 1],
            name='AttentionWeight_OriginShape')
        networkParameter['AttentionWeight_Tile'] = tensorflow.tile(
            input=networkParameter['AttentionWeight_OriginShape'],
            multiples=[1, 1, 1, networkParameter['HiddenNoduleNumber']], name='AttentionWeight_Tile')

        networkParameter['Data_AddAttention_Raw'] = tensorflow.multiply(
            networkParameter['DataInput'], networkParameter['AttentionWeight_Tile'], name='Data_AddAttention_Raw')
        networkParameter['Data_AddAttention_ReduceSum'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['Data_AddAttention_Raw'], axis=2, name='Data_AddAttention_ReduceSum')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['Data_AddAttention_ReduceSum'], axis=1, name='FinalResult')
        networkParameter['FinalResult'].set_shape([None, hiddenNoduleNumber])

    return networkParameter
