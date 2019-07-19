import numpy
import os
import shutil

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AVEC2017/'
    for foldname in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename.find('txt') == -1: continue
            with open(os.path.join(loadpath, foldname, filename), 'r') as file:
                data = file.read()
            if data.find('#') != -1:
                print(foldname, filename)
                shutil.copy(os.path.join(loadpath, foldname, filename),
                            os.path.join(loadpath, foldname, filename.replace('.txt', '-Copy.txt')))
                with open(os.path.join(loadpath, foldname, filename.replace('.txt', '-Copy.txt')), 'r') as file:
                    data = file.readlines()

                with open(os.path.join(loadpath, foldname, filename), 'w') as file:
                    for sample in data:
                        if sample.find('#') != -1: continue
                        file.write(sample)
