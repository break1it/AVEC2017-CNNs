import numpy
import os

MAX_SENTENCE = 128
MAX_LENGTH = 1000

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Step4_Normalization_Part2/'
    savepath = 'D:/PythonProjects_Data/Data_AVEC2017_HBLSTM/'

    for foldX in os.listdir(loadpath)[2:3]:
        # os.makedirs(os.path.join(savepath, foldX))
        for foldY in os.listdir(os.path.join(loadpath, foldX))[60:]:
            partData, partSeq = [], []

            if os.path.exists(os.path.join(savepath, foldX, foldY + '_Data.npy')): continue

            for filename in os.listdir(os.path.join(loadpath, foldX, foldY)):
                data = numpy.genfromtxt(fname=os.path.join(loadpath, foldX, foldY, filename), dtype=float,
                                        delimiter=',')
                partSeq.append(min(MAX_LENGTH, numpy.shape(data)[0]))

                if numpy.shape(data)[0] >= MAX_LENGTH:
                    data = data[0:MAX_LENGTH]
                else:
                    data = numpy.concatenate(
                        [data, numpy.zeros([MAX_LENGTH - numpy.shape(data)[0], numpy.shape(data)[1]])], axis=0)
                partData.append(data)
            partData, partSeq = partData[0:MAX_SENTENCE], partSeq[0:MAX_SENTENCE]

            print(foldX, foldY, numpy.shape(partData), numpy.shape(partSeq))

            numpy.save(file=os.path.join(savepath, foldX, foldY + '_Data.npy'), arr=partData)
            numpy.save(file=os.path.join(savepath, foldX, foldY + '_Seq.npy'), arr=partSeq)
