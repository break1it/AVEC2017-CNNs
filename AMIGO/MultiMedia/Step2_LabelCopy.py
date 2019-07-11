import shutil
import os

if __name__ == '__main__':
    loadpath = '/mnt/external/Bobs/AMIGOS/Experiment-Result-BLSTM2Layer/'
    savepath = '/mnt/external/Bobs/AMIGOS/Experiment-Result-BLSTM2Layer-Step2-Label/'
    os.makedirs(savepath)
    for foldname in os.listdir(loadpath):
        if foldname.find('MiddleResult') == -1: continue
        print(foldname)
        shutil.copy(os.path.join(loadpath, foldname, 'Train-0000-Label.csv'),
                    os.path.join(savepath, foldname + '-Train-Label.csv'))
        shutil.copy(os.path.join(loadpath, foldname, 'Test-0000-Label.csv'),
                    os.path.join(savepath, foldname + '-Test-Label.csv'))
    print('Completed')