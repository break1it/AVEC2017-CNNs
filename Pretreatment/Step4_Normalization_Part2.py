import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Step4_Normalization_Part1/'
    comparepath = 'D:/PythonProjects_Data/AVEC2017_Data/Step3_Spectrum/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/Step4_Normalization_Part2/'

    totalData = []
    for fold in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, fold)):
            data = numpy.load(os.path.join(loadpath, fold, filename))
            totalData.extend(data)
            print(fold, filename, numpy.shape(totalData))

    totalData = scale(totalData)

    ##########################################

    startPosition = 0
    for foldX in os.listdir(comparepath):
        for foldY in os.listdir(os.path.join(comparepath, foldX)):
            print('Writing', foldX, foldY, startPosition)
            os.makedirs(os.path.join(savepath, foldX, foldY))
            for filename in os.listdir(os.path.join(comparepath, foldX, foldY)):
                if filename == 'Transcription.csv': continue
                data = numpy.genfromtxt(fname=os.path.join(comparepath, foldX, foldY, filename), dtype=float,
                                        delimiter=',')

                writeData = totalData[startPosition:startPosition + numpy.shape(data)[0]]
                # print(numpy.shape(writeData))
                with open(os.path.join(savepath, foldX, foldY, filename), 'w') as file:
                    for indexX in range(numpy.shape(writeData)[0]):
                        for indexY in range(numpy.shape(writeData)[1]):
                            if indexY != 0: file.write(',')
                            file.write(str(writeData[indexX][indexY]))
                        file.write('\n')

                startPosition += numpy.shape(data)[0]
    print(numpy.shape(totalData), startPosition)
