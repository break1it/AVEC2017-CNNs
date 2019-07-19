import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step1_COVAREP/'
    savepath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step3_COVAREP_Assembly/'
    os.makedirs(savepath)

    ususalShape = 0
    for foldname in os.listdir(loadpath):
        totalData = []
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename.find('Participant') == -1: continue

            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            if ususalShape == 0:
                ususalShape = numpy.shape(data)[1]
            else:
                data = numpy.reshape(data, [-1, ususalShape])
            totalData.extend(data)
        print(foldname, numpy.shape(totalData))

        numpy.save(savepath + foldname + '.npy', totalData, allow_pickle=True)
