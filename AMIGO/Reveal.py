import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/AMIGO/Experiment-Result/Sample-02-Valence/Loss-%04d.csv'

    totalData = []
    for episode in range(100):
        data = numpy.genfromtxt(fname=loadpath % episode, dtype=float, delimiter=',')
        totalData.append(numpy.average(data))
    plt.plot(totalData)
    plt.xlabel('Train Episode')
    plt.ylabel('Loss')
    plt.show()
