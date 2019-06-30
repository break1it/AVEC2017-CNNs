import os
import numpy
from Auxiliary.Loader import Loader_CNN
from Model.SimpleCNN import SimpleCNN

if __name__ == '__main__':
    trainData, trainLabel, developData, developLabel, testData, testLabel = Loader_CNN(maxSentence=10)

    savepath = 'D:/PythonProjects_Data/Experiment/SimpleCNN/'
    os.makedirs(savepath)

    classifier = SimpleCNN(trainData=numpy.concatenate([trainData, developData], axis=0),
                           trainLabel=numpy.concatenate([trainLabel, developLabel], axis=0))
    for episode in range(100):
        print('\nTrain Episode %d Total Loss = %f' % (episode, classifier.Train(logName=savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + 'Network-%04d' % episode)
        classifier.Test(logName=savepath + 'TestResult-%04d.csv' % episode, testData=testData, testLabel=testLabel)
