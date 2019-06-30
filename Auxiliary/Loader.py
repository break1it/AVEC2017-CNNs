import numpy
import os


def Loader_CNN(maxSentence=5):
    loadPath = 'D:/PythonProjects_Data/Data_AVEC2017/CNN/'
    labelPath = 'D:/PythonProjects_Data/Data_AVEC2017/'

    trainData, trainLabel, developData, developLabel, testData, testLabel = [], [], [], [], [], []

    for loadPart in ['train', 'dev', 'test']:
        labelData = numpy.genfromtxt(fname=os.path.join(labelPath, '%sLabel.csv' % loadPart), dtype=int,
                                     delimiter=',')
        for searchIndex in range(min(len(labelData), maxSentence)):
            batchData = numpy.load(file=os.path.join(loadPath, loadPart, '%d_P.npy' % labelData[searchIndex][0]))
            print('Loading', loadPart, labelData[searchIndex][0], numpy.shape(batchData))

            if numpy.shape(batchData)[0] < 128:
                batchData = numpy.concatenate([batchData, numpy.zeros(
                    [128 - numpy.shape(batchData)[0], numpy.shape(batchData)[1], numpy.shape(batchData)[2]])], axis=0)

            if loadPart == 'train':
                trainData.append(batchData)
                trainLabel.append(labelData[searchIndex][2])
            if loadPart == 'dev':
                developData.append(batchData)
                developLabel.append(labelData[searchIndex][2])
            if loadPart == 'test':
                testData.append(batchData)
                testLabel.append(labelData[searchIndex][2])

    print(numpy.shape(trainData), numpy.shape(trainLabel))
    print(numpy.shape(developData), numpy.shape(developLabel))
    print(numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, developData, developLabel, testData, testLabel


if __name__ == '__main__':
    Loader_CNN()
