import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.SimpleCNN import SimpleCNN_3Layers

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = \
        Loader_CNN(partName='CNN-10')

    savepath = 'D:/PythonProjects_Data/Experiment/SimpleCNN-10-Layer3/'
    os.makedirs(savepath)

    classifier = SimpleCNN_3Layers(trainData=numpy.concatenate([trainData, developData], axis=0),
                                   trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0))
    for episode in range(100):
        print(
            '\nTrain Episode %d Total Loss = %f' % (episode, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
        classifier.Test(logName=savepath + 'TrainResult-%04d.csv' % episode, testData=trainData, testLabel=trainLabel)
        classifier.Test(logName=savepath + 'TestResult-%04d.csv' % episode, testData=testData, testLabel=testLabel)
