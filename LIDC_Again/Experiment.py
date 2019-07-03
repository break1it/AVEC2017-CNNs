from LIDC_Again.Model import StackingEnsemble
from LIDC_Again.Loader import Loader
import os
import multiprocessing as mp
import time


def Treatment():
    for partName in ['Curvelet', 'DicFeature', 'LBP_R=1_P=4', 'LBP_R=1_P=8', 'LBP_R=2_P=16', 'LBP_R=3_P=24']:
        for partType in ['DX', 'PCA']:
            for maxScope in range(1, 31):
                for testAppoint in range(5):
                    savepath = 'E:/ProjectData_LIDC/Result/%s_%s_%s/Scope=%d_Part=%d/' % (
                        partName, partType, classifierType, maxScope, testAppoint)
                    if os.path.exists(savepath): continue
                    trainData, trainLabel, testData, testLabel = Loader(partName=partName, partType=partType,
                                                                        testAppoint=testAppoint, maxScope=maxScope)

                    classifier = StackingEnsemble(
                        trainData=trainData, trainLabel=trainLabel, testData=testData, testLabel=testLabel,
                        classifierType=classifierType, savepath=savepath, splitTimes=splitTimes)


splitTimes = 5
classifierType = 'Tree'
if __name__ == '__main__':
    # partName = 'Curvelet'
    # partType = 'DX'
    # Tree, SVC, Gaussian, AdaBoost

    processList = []
    for _ in range(1):
        p1 = mp.Process(target=Treatment)
        p1.start()
        processList.append(p1)
        time.sleep(5)
    for process in processList:
        process.join()
