import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = r'D:\PythonProjects_Data\Experiment\SimpleCNN-10-Layer3'
    totalData = []
    for index in range(100):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, '%04d.csv' % index), dtype=float, delimiter=',')
        totalData.append(numpy.average(data))
    plt.plot(totalData)
    plt.xlabel('Train Episode')
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.show()
