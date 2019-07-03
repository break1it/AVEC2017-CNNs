import numpy
import os


def Loader(partName, partType, testAppoint, maxScope=-1):
    loadpath = 'E:/ProjectData_LIDC/Feature/%s_%s/' % (partName, partType)

    trainData, trainLabel, testData, testLabel = [], [], [], []
    for index in range(5):
        currentData = numpy.load(file=os.path.join(loadpath, 'Part%d-Data.npy' % index))
        if maxScope != -1: currentData = currentData[:, 0:maxScope]
        currentLabel = numpy.genfromtxt(fname=os.path.join(loadpath, 'Featurelabel_%d.csv' % index), dtype=float,
                                        delimiter=',')
        print('Loading', numpy.shape(currentData), numpy.shape(currentLabel))
        if index == testAppoint:
            testData.extend(currentData)
            testLabel.extend(currentLabel)
        else:
            trainData.extend(currentData)
            trainLabel.extend(currentLabel)
    print('Load Completed')
    print('Train Part', numpy.shape(trainData), numpy.shape(trainLabel))
    print('Test Part', numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel


def EnsembleLoader(partName, scope, appoint):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    loadpath = 'E:/ProjectData_LIDC/Result-Assembly/'

    for part in partName:
        batchTrainData = numpy.genfromtxt(
            fname=os.path.join(loadpath, part, 'Scope=%d_Part=%d' % (scope, appoint), 'TrainData.csv'), dtype=float,
            delimiter=',')
        trainLabel = numpy.genfromtxt(
            fname=os.path.join(loadpath, part, 'Scope=%d_Part=%d' % (scope, appoint), 'TrainLabel.csv'),
            dtype=float, delimiter=',')
        batchTestData = numpy.genfromtxt(
            fname=os.path.join(loadpath, part, 'Scope=%d_Part=%d' % (scope, appoint), 'TestData.csv'), dtype=float,
            delimiter=',')
        testLabel = numpy.genfromtxt(
            fname=os.path.join(loadpath, part, 'Scope=%d_Part=%d' % (scope, appoint), 'TestLabel.csv'),
            dtype=float, delimiter=',')
        # print(numpy.shape(trainData), numpy.shape(testData), numpy.shape(batchTrainData), numpy.shape(batchTestData))
        if len(numpy.shape(trainData)) == 1:
            trainData = batchTrainData.copy()
            testData = batchTestData.copy()
        else:
            trainData = numpy.concatenate([trainData, batchTrainData], axis=1)
            testData = numpy.concatenate([testData, batchTestData], axis=1)

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel
