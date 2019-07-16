import numpy
import os
import matplotlib.pylab as plt
import random

if __name__ == '__main__':
    data = numpy.genfromtxt(fname=r'D:\Workings\Input.csv', dtype=float, delimiter=',')
    # print(data)
    with open(r'D:\Workings\Output.csv', 'w') as file:
        for indexX in range(numpy.shape(data)[0]):
            for indexY in range(numpy.shape(data)[1]):
                if indexY != 0: file.write(',')
                current = (random.random() - 0.5)
                print(current, end='\t')
                file.write(str(data[indexX][indexY] + current) + '%')
            print()
            file.write('\n')
