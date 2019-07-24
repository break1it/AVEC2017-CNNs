import os
import numpy

MAX_SENTENCE = 128
MAX_FRAME = 1000

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Step4_Normalization_Part2/'
    savepath = 'D:/PythonProjects_Data/Data_AVEC2017/CNN-10-Seq/'
    # os.makedirs(savepath)
    for foldA in os.listdir(loadpath):
        os.makedirs(os.path.join(savepath, foldA))
        for foldB in os.listdir(os.path.join(loadpath, foldA)):
            totalData, totalSeq = [], []

            for filename in os.listdir(os.path.join(loadpath, foldA, foldB)):
                if numpy.shape(totalData)[0] >= MAX_SENTENCE: continue

                loadData = numpy.genfromtxt(fname=os.path.join(loadpath, foldA, foldB, filename), dtype=float,
                                            delimiter=',') * 10
                totalSeq.append(min(MAX_FRAME, len(loadData)))

                if numpy.shape(loadData)[0] >= MAX_FRAME:
                    currentData = loadData[0:MAX_FRAME]
                else:
                    currentData = numpy.concatenate(
                        [loadData, numpy.zeros([MAX_FRAME - numpy.shape(loadData)[0], numpy.shape(loadData)[1]])],
                        axis=0)

                # print(foldA, foldB, filename, numpy.shape(loadData), numpy.shape(currentData))
                totalData.append(currentData)

            print(foldA, foldB, numpy.shape(totalData))
            # exit()

            # numpy.save(file=os.path.join(savepath, foldA, foldB + '.npy'), arr=totalData)
            numpy.save(file=os.path.join(savepath, foldA, foldB + '-Label.npy'), arr=totalSeq)
