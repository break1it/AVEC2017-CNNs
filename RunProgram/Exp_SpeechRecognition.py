import os
from Auxiliary.Loader import Loader_SpeechRecognition
from Model.CNN_EncoderDecoder_SR import CNN_EncoderDecoder

if __name__ == '__main__':
    totalData, totalLabel = Loader_SpeechRecognition(maxSentence=99999)

    savepath = 'D:/PythonProjects_Data/Experiment/SR_CNN/'
    os.makedirs(savepath)
    classifier = CNN_EncoderDecoder(trainData=totalData, trainLabel=totalLabel, learningRate=1E-4, batchSize=64)
    for episode in range(100):
        print('\nTraining %d Total Loss = %f' % (
            episode, classifier.Train(logName=os.path.join(savepath + '%04d.csv' % episode))))
        classifier.Save(savepath=os.path.join(savepath + 'Network-%04d' % episode))
