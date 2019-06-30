import numpy
import os

dictionary = {'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5, 'AY': 6, 'B': 7, 'CH': 8, 'D': 9, 'DH': 10, 'EH': 11,
              'ER': 12, 'EY': 13, 'F': 14, 'G': 15, 'HH': 16, 'IH': 17, 'IY': 18, 'JH': 19, 'K': 20, 'L': 21, 'M': 22,
              'N': 23, 'NG': 24, 'OW': 25, 'OY': 26, 'P': 27, 'R': 28, 'S': 29, 'SH': 30, 'T': 31, 'TH': 32, 'UH': 33,
              'UW': 34, 'V': 35, 'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39}

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017_CMU_RAW/'
    savepath = 'D:/PythonProjects_Data/AVEC2017_CMU_Label/'

    for foldX in os.listdir(loadpath):
        for foldY in os.listdir(os.path.join(loadpath, foldX)):
            os.makedirs(os.path.join(savepath, foldX, foldY))
            print(foldX, foldY)
            for filename in os.listdir(os.path.join(loadpath, foldX, foldY)):
                with open(os.path.join(loadpath, foldX, foldY, filename), 'r') as file:
                    data = file.read()
                data = data.split(' ')

                with open(os.path.join(savepath, foldX, foldY, filename), 'w') as file:
                    for sample in data:
                        if sample in dictionary.keys():
                            file.write(str(dictionary[sample]) + ',')

            # exit()
