import numpy


def MAE_Calculation(label, predict):
    counter = 0
    for index in range(len(label)):
        counter += numpy.abs(label[index] - predict[index])
    return counter / len(label)


def RMSE_Calculation(label, predict):
    counter = 0
    for index in range(len(label)):
        counter += (label[index] - predict[index]) * (label[index] - predict[index])
    return numpy.sqrt(counter / len(label))


def CCC_Calculation(label, predict):
    result = 2 * numpy.corrcoef(label, predict)[0][1] * numpy.std(label) * numpy.std(predict)
    result = result / (
            numpy.var(label) + numpy.var(predict) + numpy.square(numpy.average(label) - numpy.average(predict)))
    return result


def CCC(x, y):
    x_mean = numpy.nanmean(x)
    y_mean = numpy.nanmean(y)

    covariance = numpy.nanmean((x - x_mean) * (y - y_mean))

    x_var = numpy.nanmean((x - x_mean) ** 2)
    y_var = numpy.nanmean((y - y_mean) ** 2)

    CCC = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2)

    return CCC


if __name__ == '__main__':
    print(numpy.var(numpy.arange(0, 5)))
    print(numpy.std(numpy.arange(0, 5)))
