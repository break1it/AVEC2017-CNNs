import numpy
import os
import re


def LoadDictionary():
    dictionary = {}
    with open('CMUDictionary.txt', 'r') as file:
        inputData = file.readlines()

        for sample in inputData:
            if sample.find('(') != -1: continue

            pattern = re.compile('0|1|2')
            sample = re.sub(pattern=pattern, repl='', string=sample)

            sample = sample.split('  ')
            dictionary[sample[0]] = sample[1][:-1]
    return dictionary


def Transform2CMU(inputData, dictionary):
    inputData = inputData.upper()
    pattern = re.compile('<*>')
    inputData = re.sub(pattern=pattern, repl='', string=inputData)
    pattern = re.compile('<|\'')
    inputData = re.sub(pattern=pattern, repl='', string=inputData)

    transformResult = ''
    splitData = inputData.split(' ')
    for part in splitData:
        if part in dictionary.keys():
            transformResult += dictionary[part] + ' '

    # print(inputData)
    # print(transformResult)
    # print('\n')
    return transformResult[0:-1]


if __name__ == '__main__':
    dictionary = LoadDictionary()

    loadpath = 'D:/PythonProjects_Data/AVEC2017_Data/Step2_VoiceSeparate/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_Data/CMU_Step1_RAW/'
    for foldX in os.listdir(loadpath):
        for foldY in os.listdir(os.path.join(loadpath, foldX)):
            os.makedirs(os.path.join(savepath, foldX, foldY))

            fileData = numpy.genfromtxt(fname=os.path.join(loadpath, foldX, foldY, 'Transcription.csv'), dtype=str,
                                        delimiter=',')
            print(foldX, foldY)
            for searchIndex in range(1, numpy.shape(fileData)[0]):
                if fileData[searchIndex][3] == 'Ellie': continue
                treatData = fileData[searchIndex][-1]
                result = Transform2CMU(inputData=treatData, dictionary=dictionary)

                with open(os.path.join(savepath, foldX, foldY, 'Speech_%s.csv' % fileData[searchIndex][0]),
                          'w') as file:
                    file.write(result)
            # print(fileData)
            # exit()
