import tensorflow


def CNN_LocalAttentionInitializer(inputData, scopeName, hiddenNoduleNumbers):
    with tensorflow.name_scope(scopeName):
        networkParameter = {}

        networkParameter['DataInput'] = inputData
        networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], \
        networkParameter['HiddenNoduleNumber'] = tensorflow.unstack(
            tensorflow.shape(networkParameter['DataInput'], name='Shape'))

        networkParameter['AttentionWeight_Flat'] = tensorflow.layers.dense(
            inputs=networkParameter['DataInput'], units=1, activation=tensorflow.nn.tanh, name='AttentionWeight_Flat')
        networkParameter['AttentionWeight_Reshape'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight_Flat'],
            shape=[networkParameter['BatchSize'], networkParameter['XScope'] * networkParameter['YScope']],
            name='AttentionWeight_Reshape')
        networkParameter['AttentionWeight_SoftMax'] = tensorflow.nn.softmax(
            logits=networkParameter['AttentionWeight_Reshape'], name='AttentionWeight_SoftMax')
        networkParameter['AttentionWeight_OriginSize'] = tensorflow.reshape(
            tensor=networkParameter['AttentionWeight_SoftMax'],
            shape=[networkParameter['BatchSize'], networkParameter['XScope'], networkParameter['YScope'], 1],
            name='AttentionWeight_OriginSize')
        networkParameter['AttentionWeight_Tile'] = tensorflow.tile(
            input=networkParameter['AttentionWeight_OriginSize'], multiples=[1, 1, 1, hiddenNoduleNumbers],
            name='AttentionWeight_Tile')

        networkParameter['Result_MultiplyAttention'] = tensorflow.multiply(
            networkParameter['DataInput'], networkParameter['AttentionWeight_Tile'], name='Result_MultiplyAttention')
        networkParameter['AssemblyMedia'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['Result_MultiplyAttention'], axis=2, name='AssemblyMedia')
        networkParameter['FinalResult'] = tensorflow.reduce_sum(
            input_tensor=networkParameter['AssemblyMedia'], axis=1, name='FinalResult')

    return networkParameter
