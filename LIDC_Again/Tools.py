import numpy
from sklearn import metrics


def Analysis(label, predict):
    matrix = numpy.zeros([2, 2])

    for index in range(numpy.shape(predict)[0]):
        matrix[numpy.argmax(predict[index])][int(label[index]) - 1] += 1

    precision = (matrix[0][0] + matrix[1][1]) / numpy.sum(matrix)
    sensitivity = matrix[0][0] / numpy.sum(matrix[0])
    specificity = matrix[1][1] / numpy.sum(matrix[1])

    totalPredict = numpy.array(predict)
    fpr, tpr, thresholds = metrics.roc_curve(label, totalPredict[:, 0], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return precision, sensitivity, specificity, auc
