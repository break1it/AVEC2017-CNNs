import numpy

if __name__ == '__main__':
    data1 = numpy.genfromtxt(
        fname=r'D:\PythonProjects_Data\LIDC-Result\Result-Single\Curvelet_DX_SVC\Scope=1_Part=0\TestPart-0-Label.csv',
        dtype=float, delimiter=',')
    data2 = numpy.genfromtxt(
        fname=r'D:\PythonProjects_Data\LIDC-Result\Result-Single\Curvelet_DX_SVC\Scope=1_Part=0\TestPart-1-Label.csv',
        dtype=float, delimiter=',')
    print(numpy.shape(data1))
    for indexX in range(numpy.shape(data1)[0]):
        if data1[indexX] != data2[indexX]: print('ERROR')
