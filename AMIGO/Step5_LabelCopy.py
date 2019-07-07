import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step1_Convert2Matrix/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step5_LabelCopy/'

    for foldname in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename.find('Label') == -1: continue
            print('Treating', foldname, filename)

            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')

            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for indexX in range(numpy.shape(data)[0]):
                    for indexY in range(1, numpy.shape(data)[1]):
                        if indexY != 1: file.write(',')
                        if data[indexX][indexY] < 0:
                            file.write('0')
                        else:
                            file.write('1')
                    file.write('\n')
