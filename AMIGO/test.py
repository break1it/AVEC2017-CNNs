from AMIGO.Loader import Loader
import numpy

if __name__ == '__main__':
    trainData, trainLabel, testData, testLabel = Loader(name='Vector00', appoint=0)

    counter = 0
    for index in range(len(trainLabel)):
        if trainLabel[index][0] == 1 and trainLabel[index][1] == 0:
            counter += 1
    print(counter)
    # exit()
