import os
import numpy
from AMIGO.Loader import Loader
from AMIGO.Model_CNN import CNN_SingleTask


def LabelPretreatment(label):
    returnLabel = []
    for sample in label:
        if sample == 0:
            returnLabel.append([1, 0])
        else:
            returnLabel.append([0, 1])
    return returnLabel


if __name__ == '__main__':
    appoint = 2
    axis = 0

    savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result/Sample-%02d-%s' % (
        appoint, ['Valence', 'Arousal'][axis])

    os.makedirs(savepath)
    os.makedirs(savepath + '-TestResult')

    trainData, trainLabel, testData, testLabel = Loader(name='Vector00', appoint=appoint)
    trainLabel = LabelPretreatment(trainLabel[:, axis])

    # print(trainLabel)
    classifier = CNN_SingleTask(trainData=trainData, trainLabel=trainLabel, learningRate=1E-3)
    for episode in range(100):
        print('\nTrain Episode %d Total Loss = %f' % (
            episode, classifier.Train_Inbalance(logName=savepath + '/Loss-%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '/Network-%04d' % episode)
        classifier.Test(logName=savepath + '-TestResult/Predict-%04d.csv' % episode, testData=trainData,
                        testLabel=trainLabel)
