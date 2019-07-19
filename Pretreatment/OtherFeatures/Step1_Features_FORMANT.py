import numpy
import os

if __name__ == '__main__':
    features = 'COVAREP'
    loadpath = 'D:/PythonProjects_Data/AVEC2017/'
    savepath = 'D:/PythonProjects_Data/AVEC2017-OtherFeatures/Step1_%s/' % features
    for foldname in os.listdir(loadpath):
        if foldname.find('_P') == -1: continue
        if os.path.exists(os.path.join(savepath, foldname)): continue
        os.makedirs(os.path.join(savepath, foldname))
        print('Treating', foldname)

        transcriptData = numpy.genfromtxt(
            fname=os.path.join(loadpath, foldname, '%s_TRANSCRIPT.csv' % foldname[0:foldname.find('_')]), dtype=str,
            delimiter='\t')
        originData = numpy.genfromtxt(
            fname=os.path.join(loadpath, foldname, '%s_%s.csv' % (foldname[0:foldname.find('_')], features)),
            dtype=str, delimiter=',')

        for index in range(1, numpy.shape(transcriptData)[0]):
            startPosition, endPosition = float(transcriptData[index][0]), float(transcriptData[index][1])

            with open(os.path.join(savepath, foldname, '%s_%04d.csv' % (transcriptData[index][2], index)), 'w') as file:
                batchData = originData[int(startPosition * 100):int(endPosition * 100)]
                if len(batchData) == 0: continue
                for indexX in range(numpy.shape(batchData)[0]):
                    for indexY in range(numpy.shape(batchData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(batchData[indexX][indexY]))
                    file.write('\n')
        # exit()
