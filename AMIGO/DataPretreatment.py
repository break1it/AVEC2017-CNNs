# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 01:43:47 2019

@author: superlee
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:04:52 2019

@author: superlee
"""

# AMIGOS数据预处理
'''
AMIGOS数据集共采集了40个被试的20个视频（16短，4长），采样率128Hz
No.8,24,28 未参加长视频录制
外部专家标注
每段视频包含5s预实验+视频观看时长
视频切分：20s一帧
第一帧：5s预实验+15s视频
第二帧以后：从5s开始，每20s取一帧，无帧叠
最后一帧：视频最后20s
所有人可切分为340帧（94短视频，246长视频），与外部标注相对应
'''
import math
import scipy.io as sio
import scipy
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
import time
import os
import pylab

# coding=gbk
from PIL import Image


# import scipy

def ImageToMatrix(filename):  # 读取图片
    im = Image.open(filename)

    width, height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.array(data, dtype='float') / 255.0
    # new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data, (height, width))
    return new_data


#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


label = np.empty((0, 2))
raw_subject_index = np.arange(0, 40, 1)
delete_index = [0, 7, 23, 27]
subject_index = np.delete(raw_subject_index, delete_index)
subject_index1 = subject_index[:]
# index
ctr = 0
part_vect = np.zeros((1, 12580), dtype=int)
trial_vect = np.zeros((1, 12580), dtype=int)
index = np.zeros((1, 12580), dtype=int)
for i in subject_index1:
    start = time.clock()
    # exec('s{}_raw_data=np.empty((0,0,17,340))'.format(i)) # 存储每个被试17个通道的频谱图
    load_fn = 'D:/PythonProjects_Data/raw_data/Data_Preprocessed_P%0.2d/Data_Preprocessed_P%0.2d.mat' % (
        i + 1, i + 1)  # 读取每个人的文件
    print('读取第%d个人的数据和标签...' % (i + 1))
    load_data = sio.loadmat(load_fn)
    joined_data = load_data['joined_data']  # 20个cell，每个cell格式：采样点*通道数
    labels_ext_annotation = load_data['labels_ext_annotation']
    labels_selfassessment = load_data['labels_selfassessment']
    VideoIDs = load_data['VideoIDs']
    subject_raw_data = np.empty((2560, 17, 0))
    for j in range(20):
        print('读取第%d个视频...' % (j + 1))
        num_frame = labels_ext_annotation[0, j].shape[0]  # 每个视频分帧的数量
        # 每个视频切分的样本
        temp = joined_data[0, j]  # 每个视频的数据
        temp_label = labels_ext_annotation[0, j][:, 1:]  # 每个视频的标签
        label = np.append(label, temp_label, 0)  # 第i个人所有视频标签初始化
        # 每个人原始数据样本排列
        video_raw_data = np.zeros((20 * 128, 17, num_frame))  # 每个视频切分的样本
        video_raw_data[:, :, 0] = temp[:20 * 128, :]  # first_frame
        video_raw_data[-2560:-1, :, -1]  # last_frame
        for k in range(num_frame - 2):
            video_raw_data[:, :, k + 1] = temp[5 * (k + 1) * 128:(20 + 5 * (k + 1)) * 128, :]

        subject_raw_data = np.append(subject_raw_data, video_raw_data, 2)  # 收集第i个人所有原始数据样本
        part_vect[0, ctr] = i
        trial_vect[0, ctr] = j
        index[0, ctr] = ctr
        ctr = ctr + 1
    # exec('s{}_raw_data=subject_raw_data'.format(i+1))  # 存储第i个人所有原始数据样本
    print('提取每个人的频谱图，设置AMIGOS频谱图数据格式...')
    fs = 128
    window = 256
    overlap = 0
    num_samples = subject_raw_data.shape[2]
    channels = subject_raw_data.shape[1]
    spectrogram_instance = np.zeros((340, 224, 341, 17))
    for m in range(num_samples):
        #    print('读取第%d个样本...' %(i+1))
        for n in range(channels):
            print('读取第%d个人第%d个样本的第%d个信道...' % (i + 1, m + 1, n + 1))
            plt.specgram(subject_raw_data[:, n, m], NFFT=window, Fs=fs, noverlap=overlap)
            plt.axis('off')
            plt.axes().get_xaxis().set_visible(False)
            plt.axes().get_yaxis().set_visible(False)
            fig = matplotlib.pyplot.gcf()
            #            if "."in filename:
            #               Filename = filename.split(".")[0]
            plt.savefig(
                'D:/PythonProjects_Data/raw_data_processing/AMIGOS_image/%0.5d-%0.2d.jpg' % (i * 340 + m, n),
                bbox_inches='tight', pad_inches=0)
            plt.clf()
    #            spectrogram_instance[m,:,:,n]=ImageToMatrix('testtest.jpg')
    #            os.system("pause")
    # del im
    # exec('s{}_spectrogram_instance=spectrogram_instance'.format(i+1))  # 存储第i个人所有频谱图数据样本
    #    np.savez('F:/AMIGOS/spectrogram/s%0.2d_spectrogram_instances1.npz' %(i+1),spectrogram_instance)
    print('数据保存成功...')
    # 清理变量
    print('清理变量')

    elapsed = (time.clock() - start)
    print('Time used: ', elapsed / 60, 's...')
