import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step1_features/'

    for foldname in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            if len(data) == 0:
                os.remove(os.path.join(loadpath, foldname, filename))
                print(foldname, filename)
