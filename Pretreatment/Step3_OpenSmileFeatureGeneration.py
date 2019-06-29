import os


def OpenSmileCall_Single(loadpath, confPath, savepath):
    commandPath = r'D:\PythonProjects\opensmile-2.3.0\bin\Win32\SMILExtract_Release'
    confPath = 'D:\\PythonProjects\\opensmile-2.3.0\\config\\' + confPath
    os.system(commandPath + ' -C ' + confPath + ' -I ' + loadpath + ' -O ' + savepath + '.current')

    loadfile = open(savepath + '.current', 'r')
    data = loadfile.readlines()
    loadfile.close()

    file = open(savepath, 'w')
    for sample in data:
        if sample[0] == '@': continue
        if len(sample) < 5: continue
        file.write(sample[sample.find(',') + 1:-3] + '\n')
    file.close()
    os.remove(savepath + '.current')


if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_Step2_VoiceSeparate/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Step3_OpenSmileFeatures/'

    # confList = ['ComParE', 'IS09', 'IS10', 'IS13', 'eGeMAPSv01a', 'GeMAPSv01a']
    # confPath = ['ComParE_2016.conf', 'IS09_emotion.conf', 'IS10_paraling.conf', 'IS13_ComParE.conf',
    #             'gemaps\\eGeMAPSv01a.conf', 'gemaps\\GeMAPSv01a.conf']
    confList = 'IS09'
    confPath = 'IS09_emotion.conf'

    for foldX in os.listdir(loadpath)[2:3]:
        for foldY in os.listdir(os.path.join(loadpath, foldX))[100:]:
            os.makedirs(os.path.join(savepath, confList, foldX, foldY))

            for filename in os.listdir(os.path.join(loadpath, foldX, foldY)):
                if filename[-3:] != 'wav': continue
                print('Treating', foldX, foldY, filename)
                OpenSmileCall_Single(loadpath=os.path.join(loadpath, foldX, foldY, filename), confPath=confPath,
                                     savepath=os.path.join(savepath, confList, foldX, foldY, filename + '.csv'))
                # exit()
