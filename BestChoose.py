from Auxiliary.Tools import MAE_Calculation, RMSE_Calculation, CCC_Calculation
import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/Experiment/SimpleCNN-10-Layer3-TestResult/'

    MAEList, RMSEList = [], []
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')
        MAEList.append(MAE_Calculation(label=data[:, 0], predict=data[:, 1]))
        RMSEList.append(RMSE_Calculation(label=data[:, 0], predict=data[:, 1]))
    print('MAE = %.2f\tRMSE = %.2f' % (min(MAEList), min(RMSEList)))

    plt.plot(MAEList)
    plt.plot(RMSEList)
    plt.show()
