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


def Loader_SpeechRecognition(maxSentence=10):
    loadpath = 'D:/PythonProjects_Data/Data_AVEC2017_SpeechRecognition/'
    totalData, totalLabel = [], []

    for fold in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, fold))[0:maxSentence]:
            if filename.find('Data') == -1: continue

            batchData = numpy.load(file=os.path.join(loadpath, fold, filename))
            with open(os.path.join(loadpath, fold, filename.replace('Data.npy', 'Label.csv')), 'r') as file:
                batchLabelStr = file.readlines()
                batchLabel = []
                for sample in batchLabelStr:
                    sample = sample.split(',')
                    current = []
                    for subsample in sample[0:-1]:
                        current.append(int(subsample))
                    batchLabel.append(current)

            print(fold, filename, numpy.shape(batchData), numpy.shape(batchLabel))
            totalData.extend(batchData)
            totalLabel.extend(batchLabel)

    print(numpy.shape(totalData), numpy.shape(totalLabel))
    return totalData, totalLabel


if __name__ == '__main__':
    Loader_SpeechRecognition()
