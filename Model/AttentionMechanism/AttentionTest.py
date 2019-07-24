from Model.Base import NeuralNetwork_Base
import tensorflow
import numpy


class AttentionTest(NeuralNetwork_Base):
    def __init__(self):
        super(AttentionTest, self).__init__(trainData=None, trainLabel=None)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, None], name='dataInput')
        self.maskInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='maskInput')
        self.result = (tensorflow.sequence_mask(lengths=self.maskInput, maxlen=10, dtype=tensorflow.int32,
                                                name='') * 2 - tensorflow.ones(shape=[3, 10],
                                                                               dtype=tensorflow.int32)) * 9999

    def Valid(self):
        result = self.session.run(fetches=self.result,
                                  feed_dict={self.maskInput: [3, 4, 5]})
        print(numpy.shape(result))
        print(result)


if __name__ == '__main__':
    classifier = AttentionTest()
    classifier.Valid()
