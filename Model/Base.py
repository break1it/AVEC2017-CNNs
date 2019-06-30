import tensorflow


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
