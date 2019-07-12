import numpy
import os

if __name__ == '__main__':
    for appoint in range(2, 17, 2):
        datapath = 'D:/PythonProjects_Data/AMIGO/Step4_Split/Vector%02d/' % appoint
        labelpath = 'D:/PythonProjects_Data/AMIGO/Step5_LabelCopy/'
        savepath = 'D:/PythonProjects_Data/AMIGO/Step6_FinalNpy/Vector%02d/' % appoint
        os.makedirs(savepath)

        for foldname in os.listdir(datapath):
            totalData, totalLabel = [], []
            for sampleIndex in range(20):
                labelData = numpy.genfromtxt(
                    fname=os.path.join(labelpath, foldname, 'Sample%02d-Label.csv' % sampleIndex), dtype=int,
                    delimiter=',')

                for subsampleIndex in range(numpy.shape(labelData)[0]):
                    data = numpy.genfromtxt(
                        fname=os.path.join(datapath, foldname, 'Sample%02d-%02d.csv' % (sampleIndex, subsampleIndex)),
                        dtype=float, delimiter=',')
                    totalData.append(data)
                    totalLabel.append(labelData[subsampleIndex])

            print(foldname, numpy.shape(totalData), numpy.shape(totalLabel))

            numpy.save(file=os.path.join(savepath, foldname + '_Data.npy'), arr=totalData)
            numpy.save(file=os.path.join(savepath, foldname + '_Label.npy'), arr=totalLabel)

            # exit()
