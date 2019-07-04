import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step3_Normalization/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step4_FinalData/'

    os.makedirs(savepath)

    for foldname in os.listdir(loadpath):
        totalData = []
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename.find('Label') != -1: continue
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            totalData.extend(data)
        print(foldname, numpy.shape(totalData))
        numpy.save(os.path.join(savepath, foldname + '.npy'), totalData)
        # exit()
