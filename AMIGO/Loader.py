import numpy
import os


def Loader(name, appoint):
    loadpath = 'D:/PythonProjects_Data/AMIGO/Data/%s/' % name

    trainData, trainLabel, testData, testLabel = [], [], [], []

    for searchIndex in range(2, 41):
        if searchIndex in [8, 9, 12, 17, 18, 21, 22, 23, 24, 28, 33]: continue

        filename = 'Data_Preprocessed_P%02d_Data.npy' % searchIndex

        data = numpy.load(os.path.join(loadpath, filename))
        label = numpy.load(os.path.join(loadpath, filename.replace('_Data', '_Label')))
        # print(numpy.shape(data), numpy.shape(label))
        print('Loading', filename)

        if searchIndex == appoint:
            testData.extend(data)
            testLabel.extend(label)
        else:
            trainData.extend(data)
            trainLabel.extend(label)

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel
