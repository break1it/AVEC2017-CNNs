import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Step3_OpenSmileFeatures/IS09/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Step4_OpenSmileFeatures_Normalization/IS09/'
    featureShape = 384

    totalData = []
    for foldX in os.listdir(loadpath):
        for foldY in os.listdir(os.path.join(loadpath, foldX)):
            print(foldX, foldY, numpy.shape(totalData))
            for filename in os.listdir(os.path.join(loadpath, foldX, foldY)):
                data = numpy.reshape(
                    numpy.genfromtxt(fname=os.path.join(loadpath, foldX, foldY, filename), dtype=float, delimiter=','),
                    newshape=[-1, featureShape])
                totalData.extend(data)
    print(numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for foldX in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldX))
        for foldY in os.listdir(os.path.join(loadpath, foldX)):
            print(foldX, foldY, startPosition)
            with open(os.path.join(savepath, foldX, foldY + '.csv'), 'w') as file:
                sentenceNumber = len(os.listdir(os.path.join(loadpath, foldX, foldY)))

                writeData = totalData[startPosition:startPosition + sentenceNumber]
                for indexX in range(numpy.shape(writeData)[0]):
                    for indexY in range(numpy.shape(writeData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(writeData[indexX][indexY]))
                    file.write('\n')
                startPosition += sentenceNumber
