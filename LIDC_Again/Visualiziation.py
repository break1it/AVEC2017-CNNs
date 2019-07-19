import numpy
import os
import matplotlib.pylab as plt
from LIDC_Again.Tools import Analysis

if __name__ == '__main__':
    loadpath = 'D:/PythonProjects_Data/LIDC-Result/Result-Single/LBP_R=3_P=24_PCA_SVC/'

    totalPrecision, totalSensitivity, totalSpecificity, totalAUC = [], [], [], []
    for scope in range(1, 31):

        currentPrecision, currentSensitivity, currentSpecificity, currentAUC = 0, 0, 0, 0
        for part in range(5):
            totalPredict, totalLabel = [], []
            for sample in range(5):
                predict = numpy.genfromtxt(
                    fname=os.path.join(loadpath, 'Scope=%d_Part=%d' % (scope, part),
                                       'TestPart-%d-Predict.csv' % sample),
                    dtype=float, delimiter=',')
                label = numpy.genfromtxt(
                    fname=os.path.join(loadpath, 'Scope=%d_Part=%d' % (scope, part),
                                       'TestPart-%d-Label.csv' % sample),
                    dtype=float, delimiter=',')
                if len(totalLabel) == 0:
                    totalLabel = label
                    totalPredict = predict
                else:
                    totalLabel += label
                    totalPredict += predict
            totalPredict /= 5
            totalLabel /= 5
            precision, sensitivity, specificity, auc = Analysis(label=totalLabel, predict=totalPredict)

            currentPrecision += precision / 5
            currentSensitivity += sensitivity / 5
            currentSpecificity += specificity / 5
            currentAUC += auc / 5
        print(scope, currentPrecision, currentSensitivity, currentSpecificity, currentAUC)
        # exit()
        totalPrecision.append(currentPrecision)
        totalSensitivity.append(currentSensitivity)
        totalSpecificity.append(currentSpecificity)
        totalAUC.append(currentAUC)
    plt.plot(totalPrecision, label='Precision')
    plt.plot(totalSensitivity, label='Sensitivity')
    plt.plot(totalSpecificity, label='Specificity')
    plt.plot(totalAUC, label='AUC')
    plt.legend()
    plt.title('LBP_R=3_P=24_PCA_SVC')
    plt.xlabel('Number of Features')
    plt.show()
