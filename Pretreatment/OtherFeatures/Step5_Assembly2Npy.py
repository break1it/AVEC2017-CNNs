import os
import numpy
import matplotlib.pylab as plt

MAX_SENTENCE = 128
MAX_FRAME = 1000

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step4_COVAREP_Normalization/'
    savepath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Data_COVAREP/'
    # os.makedirs(savepath)

    usualShape = 0

    for foldname in os.listdir(loadpath):
        totalData = []
        for filename in os.listdir(os.path.join(loadpath, foldname))[0:MAX_SENTENCE]:
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            if usualShape == 0:
                usualShape = numpy.shape(data)[1]
            else:
                data = numpy.reshape(data, [-1, usualShape])

            if numpy.shape(data)[0] < MAX_FRAME:
                data = numpy.concatenate([data, numpy.zeros([MAX_FRAME - numpy.shape(data)[0], numpy.shape(data)[1]])],
                                         axis=0)
            else:
                data = data[0:MAX_FRAME]

            totalData.append(data)
        print(foldname, numpy.shape(totalData))
        numpy.save(savepath + foldname + '.npy', totalData)
