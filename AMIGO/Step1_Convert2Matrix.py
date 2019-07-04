from scipy.io import loadmat
import numpy
import os

loadpath = 'D:/PythonProjects_Data/AMIGO/raw_data/'
savepath = 'D:/PythonProjects_Data/AMIGO/Step1_Convert2Matrix/'
for foldname in os.listdir(loadpath):
    for filename in os.listdir(os.path.join(loadpath, foldname)):
        print('Treating', filename)
        data = loadmat(os.path.join(loadpath, foldname, filename))

        os.makedirs(os.path.join(savepath, foldname))
        # print(data.keys())
        for indexA in range(len(data['joined_data'])):
            for indexB in range(len(data['joined_data'][indexA])):
                # print(numpy.shape(data['joined_data'][indexA][indexB]))

                with open(os.path.join(savepath, foldname, 'Sample%02d.csv' % indexB), 'w') as file:
                    for indexX in range(numpy.shape(data['joined_data'][indexA][indexB])[0]):
                        for indexY in range(numpy.shape(data['joined_data'][indexA][indexB])[1]):
                            if indexY != 0: file.write(',')
                            file.write(str(data['joined_data'][indexA][indexB][indexX][indexY]))
                        file.write('\n')

        for indexA in range(len(data['labels_ext_annotation'])):
            for indexB in range(len(data['labels_ext_annotation'][indexA])):
                # print(numpy.shape(data['labels_ext_annotation'][indexA][indexB]))
                with open(os.path.join(savepath, foldname, 'Sample%02d-Label.csv' % indexB), 'w') as file:
                    for indexX in range(numpy.shape(data['labels_ext_annotation'][indexA][indexB])[0]):
                        for indexY in range(numpy.shape(data['labels_ext_annotation'][indexA][indexB])[1]):
                            if indexY != 0: file.write(',')
                            file.write(str(data['labels_ext_annotation'][indexA][indexB][indexX][indexY]))
                        file.write('\n')
        # exit()
