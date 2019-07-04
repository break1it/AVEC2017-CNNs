import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    originpath = 'D:/PythonProjects_Data/AMIGO/Step1_Convert2Matrix/'
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step2_AssemblyNpy/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step3_Normalization/'

    totalData = []

    for filename in os.listdir(loadpath):
        data = numpy.load(os.path.join(loadpath, filename))
        totalData.extend(data)
        print(filename, numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for foldname in os.listdir(originpath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(originpath, foldname)):
            if filename.find('Label') != -1: continue

            data = numpy.genfromtxt(fname=os.path.join(originpath, foldname, filename), dtype=float, delimiter=',')

            saveData = totalData[startPosition:startPosition + len(data)]
            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for indexX in range(numpy.shape(saveData)[0]):
                    for indexY in range(numpy.shape(saveData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(saveData[indexX][indexY]))
                    file.write('\n')

            startPosition += len(data)
            print('Writing', foldname, filename, startPosition)
