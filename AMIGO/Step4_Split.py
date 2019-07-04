import numpy
import os


def WriteCsv(filename, data):
    with open(filename, 'w') as file:
        pass


if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step1_Convert2Matrix/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step3_Split/'

    for foldname in os.listdir(loadpath)[0:1]:
        os.makedirs(os.path.join(savepath, foldname))
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename.find('-Label') != -1: continue
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')

            star

            print(numpy.shape(data))
            exit()
