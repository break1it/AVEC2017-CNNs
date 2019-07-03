from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import multiprocessing
import numpy
import os
from sklearn.externals import joblib
import tensorflow
import random


def Shuffle(data, label):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel = [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
    return newData, newLabel


class SimpleClassifier:
    def __init__(self, trainData, trainLabel, classifierType):
        self.data, self.label = trainData, trainLabel
        if classifierType == 'SVC': self.clf = SVC(probability=True)
        if classifierType == 'Gaussian': self.clf = GaussianNB()
        if classifierType == 'Tree': self.clf = DecisionTreeClassifier()
        if classifierType == 'AdaBoost': self.clf = AdaBoostClassifier()

    def Train(self):
        self.clf.fit(self.data, self.label)

    def Test(self, testData, savename):
        predict = self.clf.predict_proba(testData)
        with open(savename, 'w') as file:
            for indexX in range(numpy.shape(predict)[0]):
                for indexY in range(numpy.shape(predict)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(predict[indexX][indexY]))
                file.write('\n')

    def Save(self, savepath):
        joblib.dump(self.clf, savepath)

    def Load(self, loadpath):
        self.clf = joblib.load(loadpath)


class StackingEnsemble:
    def __init__(self, trainData, trainLabel, testData, testLabel, classifierType, savepath, splitTimes=5,
                 multiprocessingFlag=False):
        self.data, self.label, self.splitTimes = trainData, trainLabel, splitTimes
        self.testData, self.testLabel = testData, testLabel
        self.savepath = savepath
        self.multiprocessingFlag = multiprocessingFlag

        self.classifierList = []
        for _ in range(splitTimes):
            if classifierType == 'SVC': clf = SVC(probability=True)
            if classifierType == 'Gaussian': clf = GaussianNB()
            if classifierType == 'Tree': clf = DecisionTreeClassifier()
            if classifierType == 'AdaBoost': clf = AdaBoostClassifier()
            self.classifierList.append(clf)

        if os.path.exists(savepath): return
        print('\nPretreatment Completed')
        print('Start Training...')

        os.makedirs(savepath)
        if self.multiprocessingFlag:
            pass
        else:
            self.Train_Single()

    def Train_Single(self):
        splitData, splitLabel = [], []
        for index in range(self.splitTimes):
            splitData.append(self.data[index::self.splitTimes])
            splitLabel.append(self.label[index::self.splitTimes])

        for index in range(self.splitTimes):
            batchTrainData, batchTrainLabel, batchTestData, batchTestLabel = [], [], splitData[index], splitLabel[index]
            for splitIndex in range(self.splitTimes):
                if splitIndex == index: continue
                batchTrainData.extend(splitData[splitIndex])
                batchTrainLabel.extend(splitLabel[splitIndex])

            print('Traing Split %d' % index, numpy.shape(batchTrainData), numpy.shape(batchTrainLabel),
                  numpy.shape(batchTestData), numpy.shape(batchTestLabel))

            self.classifierList[index].fit(batchTrainData, batchTrainLabel)
            joblib.dump(self.classifierList[index], self.savepath + 'BatchPredict-%d.m' % index)

            self.Test(logName='TrainPart-%d' % index, classifier=self.classifierList[index], testData=self.data,
                      testLabel=self.label)
            # self.Test(logName='DevelopPart-%d' % index, classifier=self.classifierList[index], testData=batchTestData,
            #           testLabel=batchTestLabel)
            self.Test(logName='TestPart-%d' % index, classifier=self.classifierList[index], testData=self.testData,
                      testLabel=self.testLabel)

    def Test(self, logName, classifier, testData, testLabel):
        predict = classifier.predict_proba(testData)
        with open(self.savepath + '%s-Predict.csv' % logName, 'w') as file:
            for indexX in range(numpy.shape(predict)[0]):
                for indexY in range(numpy.shape(predict)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(predict[indexX][indexY]))
                file.write('\n')
        with open(self.savepath + '%s-Label.csv' % logName, 'w') as file:
            for indexX in range(numpy.shape(testLabel)[0]):
                file.write(str(testLabel[indexX]) + '\n')

    def Load(self):
        for index in range(self.splitTimes):
            self.classifierList[index] = joblib.load(self.savepath + 'BatchPredict-%d.m' % index)


class NeuralNetwork:
    def __init__(self, trainData, trainLabel, hiddenNoduleNumber, layers=2, batchSize=32, learningRate=1E-4):
        self.data, self.label = trainData, trainLabel
        self.layers, self.hiddenNoduleNumber, self.batchSize = layers, hiddenNoduleNumber, batchSize

        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tensorflow.Session(config=config)

        self.parameters = {}
        self.BuildNetwork(learningRate=learningRate)
        self.session.run(tensorflow.global_variables_initializer())

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, numpy.shape(self.data)[1]],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 2], name='labelInput')
        self.parameters['FC_0'] = tensorflow.layers.dense(inputs=self.dataInput, units=self.hiddenNoduleNumber,
                                                          activation=tensorflow.nn.relu, name='First_0')

        for index in range(1, self.layers):
            self.parameters['FC_%d' % index] = tensorflow.layers.dense(
                inputs=self.parameters['FC_%d' % (index - 1)], units=self.hiddenNoduleNumber,
                activation=tensorflow.nn.relu, name='FC_%d' % index)

        self.parameters['Predict'] = tensorflow.layers.dense(inputs=self.parameters['FC_%d' % (self.layers - 1)],
                                                             units=2, activation=None, name='Predict')
        self.parameters['Predict_Softmax'] = tensorflow.nn.softmax(logits=self.parameters['Predict'],
                                                                   name='Predict_Softmax')
        self.parameters['Loss'] = tensorflow.losses.softmax_cross_entropy(onehot_labels=self.labelInput,
                                                                          logits=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel = Shuffle(data=self.data, label=self.label)
        startPosition, totalLoss = 0, 0.0
        while startPosition + self.batchSize < len(trainData):
            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train], feed_dict={
                self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                self.labelInput: trainLabel[startPosition:startPosition + self.batchSize]})
            # print(numpy.shape(result))
            print('\rTraining %d/%d Loss = %f' % (startPosition, len(trainData), loss), end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss

    def Test(self, testData, savepath):
        with open(savepath, 'w') as file:
            startPosition = 0
            while startPosition + self.batchSize < len(testData):
                batchPredict = self.session.run(fetches=self.parameters['Predict_Softmax'], feed_dict={
                    self.dataInput: testData[startPosition:startPosition + self.batchSize]})
                for indexX in range(numpy.shape(batchPredict)[0]):
                    for indexY in range(numpy.shape(batchPredict)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(batchPredict[indexX][indexY]))
                    file.write('\n')
                startPosition += self.batchSize
