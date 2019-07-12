import os
import numpy
import librosa


def SpectrumGeneration(savepath, data):
    result = numpy.abs(librosa.core.stft(data, n_fft=128, hop_length=128, win_length=128))
    result = numpy.transpose(result, [1, 0])
    with open(savepath, 'w') as file:
        for indexX in range(numpy.shape(result)[0]):
            for indexY in range(numpy.shape(result)[1]):
                if indexY != 0: file.write(',')
                file.write(str(result[indexX][indexY]))
            file.write('\n')


if __name__ == '__main__':
    for APPOINT in range(2, 18, 2):
        loadpath = 'D:/PythonProjects_Data/AMIGO/Step1_Convert2Matrix/'
        savepath = 'D:/PythonProjects_Data/AMIGO/Step2_SpectrumList/Vector%02d/' % APPOINT

        for foldname in os.listdir(loadpath):
            os.makedirs(os.path.join(savepath, foldname))
            for filename in os.listdir(os.path.join(loadpath, foldname)):
                if filename.find('-Label') != -1: continue
                print('Treating', foldname, filename)

                data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename),
                                        dtype=float, delimiter=',')[:, APPOINT]
                SpectrumGeneration(savepath=os.path.join(savepath, foldname, filename), data=data)

                # exit()
