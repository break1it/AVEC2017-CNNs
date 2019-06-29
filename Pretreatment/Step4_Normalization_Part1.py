import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Step3_Spectrum/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Step4_Normalization_Part1/'

    for foldA in os.listdir(loadpath)[2:3]:
        # os.makedirs(os.path.join(savepath, foldA))
        for foldB in os.listdir(os.path.join(loadpath, foldA))[70:]:

            totalData = []
            for filename in os.listdir(os.path.join(loadpath, foldA, foldB)):
                if filename == 'Transcription.csv': continue

                data = numpy.genfromtxt(fname=os.path.join(loadpath, foldA, foldB, filename), dtype=float,
                                        delimiter=',')
                totalData.extend(data)

            print(foldA, foldB, numpy.shape(totalData))
            numpy.save(file=os.path.join(savepath, foldA, foldB + '.npy'), arr=totalData)
