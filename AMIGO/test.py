import os

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Step1_Convert2Matrix/'

    for foldname in os.listdir(loadpath):
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            with open(os.path.join(loadpath, foldname, filename), 'r') as file:
                data = file.read()
            if data.find('nan') != -1: print(foldname, filename)
