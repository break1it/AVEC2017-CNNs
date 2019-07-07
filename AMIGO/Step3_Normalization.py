import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    totalData = []

    loadpath = 'D:/PythonProjects_Data/AMIGO/Step2_SpectrumList/Vector00/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step3_Normalization/Vector00/'
    for foldname in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            totalData.extend(data)
            print('Loading', foldname, filename, numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for foldname in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            writeData = totalData[startPosition:startPosition + numpy.shape(data)[0]]
            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for indexX in range(numpy.shape(writeData)[0]):
                    for indexY in range(numpy.shape(writeData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(writeData[indexX][indexY]))
                    file.write('\n')

            startPosition += numpy.shape(data)[0]
            print('Writing', foldname, filename, startPosition)
