import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/BLSTM1Layer/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Experiment-MiddleResult/BLSTM1Layer-Step1-SoftMax-Normalization/'

    for foldname in os.listdir(loadpath):
        if os.path.exists(os.path.join(savepath, foldname)): continue
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename.find('Predict') == -1: continue

            print('Treating', foldname, filename)

            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            data = numpy.exp(data)
            sumData = numpy.tile(numpy.sum(data, axis=1, keepdims=True), [1, 2])
            data = numpy.divide(data, sumData)
            # print(data)

            with open(os.path.join(savepath, foldname, filename), 'w') as file:
                for indexX in range(numpy.shape(data)[0]):
                    for indexY in range(numpy.shape(data)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(data[indexX][indexY]))
                    file.write('\n')
            # exit()
