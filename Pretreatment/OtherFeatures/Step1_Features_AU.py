import numpy
import os
import librosa

if __name__ == '__main__':
    features = 'features3D'
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
            fname=os.path.join(loadpath, foldname, '%s_CLNF_%s.txt' % (foldname[0:foldname.find('_')], features)),
            dtype=str, delimiter=',')

        position = 1
        for index in range(1, numpy.shape(transcriptData)[0]):
            startPosition, endPosition = float(transcriptData[index][0]), float(transcriptData[index][1])

            with open(os.path.join(savepath, foldname, '%s_%04d.csv' % (transcriptData[index][2], index)), 'w') as file:
                if position >= numpy.shape(originData)[0]: break
                while startPosition > float(originData[position][1]):
                    position += 1
                    if position >= numpy.shape(originData)[0]: break
                if position >= numpy.shape(originData)[0]: break

                while float(originData[position][1]) <= endPosition:
                    if originData[position][3] == 0: continue
                    for writeIndex in range(4, len(originData[position])):
                        if writeIndex != 4: file.write(',')
                        file.write(originData[position][writeIndex])
                    position += 1
                    file.write('\n')
                    if position >= numpy.shape(originData)[0]: break
                if position >= numpy.shape(originData)[0]: break

        # exit()
