from AMIGO.Loader import Loader
import numpy

if __name__ == '__main__':
    for appoint in range(2, 40):
        trainData, trainLabel, testData, testLabel = Loader(name='Vector00', appoint=appoint)
        print('\n\n', appoint, numpy.sum(testLabel, axis=0))
        # exit()
