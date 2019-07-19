import numpy
from sklearn.preprocessing import scale
import os

if __name__ == '__main__':
    part = 'COVAREP'
    loadpath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step3_%s_Assembly/' % part
    datapath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step1_%s/' % part
    savepath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step4_%s_Normalization/' % part
    totalData = []

    usualShape = 0
    for filename in os.listdir(loadpath):
        data = numpy.load(os.path.join(loadpath, filename), allow_pickle=True)

        if usualShape == 0:
            usualShape = numpy.shape(data)[1]
        else:
            data = numpy.reshape(data, [-1, usualShape])
        totalData.extend(data)
        print(filename, numpy.shape(totalData))
    totalData = scale(totalData)

    startPosition = 0
    for foldname in os.listdir(datapath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(datapath, foldname)):
            print('Writing', foldname, filename)
            if filename.find('Participant') == -1: continue

            data = numpy.reshape(
                numpy.genfromtxt(fname=os.path.join(datapath, foldname, filename), dtype=float, delimiter=','),
                [-1, numpy.shape(totalData)[1]])

            batchData = totalData[startPosition:startPosition + numpy.shape(data)[0]]
            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for indexX in range(numpy.shape(batchData)[0]):
                    for indexY in range(numpy.shape(batchData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(batchData[indexX][indexY]))
                    file.write('\n')

            startPosition += numpy.shape(data)[0]
