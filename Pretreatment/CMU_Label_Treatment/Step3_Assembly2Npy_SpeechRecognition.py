import numpy
import os

maxLen = 1000

if __name__ == '__main__':
    datapath = 'D:/PythonProjects_Data/AVEC2017_Data/Step4_Normalization_Part2/'
    labelpath = 'D:/PythonProjects_Data/AVEC2017_Data/CMU_Step2_Label/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/CMU_Step3_Assembly/'

    for foldX in os.listdir(datapath):
        os.makedirs(os.path.join(savepath, foldX))
        for foldY in os.listdir(os.path.join(datapath, foldX)):
            partData, partSeq, partLabel = [], [], []
            for filename in os.listdir(os.path.join(datapath, foldX, foldY)):
                data = numpy.genfromtxt(fname=os.path.join(datapath, foldX, foldY, filename), dtype=float,
                                        delimiter=',')
                label = numpy.genfromtxt(fname=os.path.join(labelpath, foldX, foldY, filename), dtype=int,
                                         delimiter=',')
                if len(label) == 0: continue
                partSeq.append(min(numpy.shape(data)[0], maxLen))

                if numpy.shape(data)[0] < maxLen:
                    supplementData = numpy.zeros([maxLen - numpy.shape(data)[0], numpy.shape(data)[1]])
                    data = numpy.concatenate([data, supplementData], axis=0)
                else:
                    data = data[0:maxLen]
                partData.append(data)
                partLabel.append(label)

            # print(partSeq)
            print(foldX, foldY, numpy.shape(partData))
            # print(numpy.shape(partData))

            numpy.save(file=os.path.join(savepath, foldX, foldY + '_Data.npy'), arr=partData, allow_pickle=True)
            numpy.save(file=os.path.join(savepath, foldX, foldY + '_Seq.npy'), arr=partSeq, allow_pickle=True)

            with open(os.path.join(savepath, foldX, foldY + '_Label.csv'), 'w') as file:
                for indexX in range(len(partLabel)):
                    for indexY in range(len(partLabel[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(partLabel[indexX][indexY]))
                    file.write('\n')
            # exit()
