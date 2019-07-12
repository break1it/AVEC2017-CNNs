import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step3_Split/'
    savepath = 'D:/PythonProjects_Data/AMIGO/Step4_Assembly/'
    os.makedirs(savepath)
    for foldname in os.listdir(loadpath):
        totalData = []
        for sampleIndex in range(20):
            for batchIndex in range(100):
                if not os.path.exists(os.path.join(loadpath, foldname, 'Sample%02d-%02d.csv' % (
                        sampleIndex, batchIndex))):
                    continue
                data = numpy.genfromtxt(
                    fname=os.path.join(loadpath, foldname, 'Sample%02d-%02d.csv' % (sampleIndex, batchIndex)),
                    dtype=float, delimiter=',')
                totalData.append(data)
        print(foldname, numpy.shape(totalData))
        numpy.save(savepath + foldname + '.npy', totalData)
