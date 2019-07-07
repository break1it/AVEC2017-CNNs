import numpy
import os


def WriteCsv(filename, data):
    with open(filename, 'w') as file:
        for indexX in range(numpy.shape(data)[0]):
            for indexY in range(numpy.shape(data)[1]):
                if indexY != 0: file.write(',')
                file.write(str(data[indexX][indexY]))
            file.write('\n')


if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step3_Normalization/Vector00/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step4_Split/Vector00/'

    for foldname in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            print('Split', foldname, filename)
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')

            WriteCsv(filename=os.path.join(savepath, foldname, filename.replace('.csv', '-%02d.csv' % 0)),
                     data=data[0:20])

            startPosition, uniqueIndex = 5, 1

            while startPosition + 20 < len(data):
                WriteCsv(filename=os.path.join(savepath, foldname, filename.replace('.csv', '-%02d.csv' % uniqueIndex)),
                         data=data[startPosition:startPosition + 20])
                startPosition += 20
                uniqueIndex += 1
            WriteCsv(filename=os.path.join(savepath, foldname, filename.replace('.csv', '-%02d.csv' % uniqueIndex)),
                     data=data[len(data) - 20:len(data)])

            # exit()
