import os
import numpy
import shutil

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Step1_Filter/'

    for part in ['train', 'dev', 'test']:
        searchList = numpy.genfromtxt(fname=os.path.join(loadpath, '%s_split_Depression_AVEC2017.csv' % part),
                                      dtype=str, delimiter=',')

        for index in range(1, numpy.shape(searchList)[0]):
            print(part, searchList[index][0])
            os.makedirs(os.path.join(savepath, part, '%s_P' % searchList[index][0]))
            shutil.copy(
                src=os.path.join(loadpath, '%s_P' % searchList[index][0], '%s_AUDIO.wav' % searchList[index][0]),
                dst=os.path.join(savepath, part, '%s_P' % searchList[index][0], '%s_AUDIO.wav' % searchList[index][0]))
            shutil.copy(
                src=os.path.join(loadpath, '%s_P' % searchList[index][0], '%s_TRANSCRIPT.csv' % searchList[index][0]),
                dst=os.path.join(savepath, part, '%s_P' % searchList[index][0],
                                 '%s_TRANSCRIPT.csv' % searchList[index][0]))
        # exit()
