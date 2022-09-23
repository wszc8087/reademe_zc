# --混淆矩阵制作--
#import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time, copy, os, argparse
import numpy as np
import scipy.io as sio
import json


classes = np.array(np.arange(0, 15, 1), dtype=str)


plt.rcParams['font.sans-serif'] ='SimHei'
plt.rcParams['axes.unicode_minus'] = False

# y_pred_txt_path = '../../classification/y_ptest_20_6.txt'
# y_true_txt_path = '../../classification/y_ttest_20_6.txt'

y_pred_txt_path = '../../result/f6_e200_cbam/6/y_p_.txt'
y_true_txt_path = '../../result/f6_e200_cbam/6/y_t.txt'

y_true =np.array(np.loadtxt(y_true_txt_path), dtype=int).reshape(-1)
y_pred = np.array(np.loadtxt(y_pred_txt_path), dtype=int).reshape(-1)
# 读取数据预测类别y_pred

# 计算混淆矩阵，FP，FN，TP，TN，PRECISION，RECALL
cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

FP = sum(cm.sum(axis=0)) - sum(np.diag(cm))  # 假正样本数
FN = sum(cm.sum(axis=1)) - sum(np.diag(cm))  # 假负样本数
TP = sum(np.diag(cm))  # 真正样本数
TN = sum(cm.sum().flatten()) - (FP + FN + TP)  # 真负样本数
SUM = TP + FP
PRECISION = TP / (TP + FP)  # 查准率，又名准确率
RECALL = TP / (TP + FN)  # 查全率，又名召回率


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):  # 绘制混淆矩阵
    # plt.figure(figsize=(24, 16), dpi=100)
    plt.figure(num=None, figsize=(5, 5), dpi=600)
    #figure(num=None, figsize=(2.8, 1.7), dpi=300)
    # figsize的2.8和1.7指的是英寸，dpi指定图片分辨率。那么图片就是（2.8*300）*（1.7*300）像素大小
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes) + 1)
    x, y = np.meshgrid(ind_array, ind_array)  # 生成坐标矩阵
    diags = np.diag(cm)  # 对角TP值

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    TP_FNs, TP_FPs = [], []
    x_name = ['空物','纸巾包','瓶子','石头','球体','易拉罐','内六角','木板','鼠标','遥控器','剪刀','万用表','卷尺','海绵','胶布']

    #plt.text(len(classes), len(classes), str('%.2f' % (PRECISION * 100,)) + '%', color='red', va='center', ha='center', fontsize=7)   # 绘制右下角的准确率

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        max_index = len(classes)
        if x_val != max_index and y_val != max_index:  # 绘制混淆矩阵各格数值
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, r'{0:.2f}'.format(c), color='black', fontsize=7, va='center', ha='center')
        elif x_val == max_index and y_val != max_index:  # 绘制最右列即各数据类别的查全率
            TP = diags[y_val]
            TP_FN = cm.sum(axis=1)[y_val]
            recall = TP / (TP_FN)
            if recall != 0.0 and recall > 0.01:
                recall = str('%.2f' % (recall * 100,))  # + '%'
            elif recall == 0.0:
                recall = '0'
            TP_FNs.append(TP_FN)
            #plt.text(x_val, y_val, str(TP_FN) + '\n' + str(recall), fontsize=7, color='black', va='center', ha='center')
        elif x_val != max_index and y_val == max_index:  # 绘制最下行即各数据类别的查准率
            TP = diags[x_val]
            TP_FP = cm.sum(axis=0)[x_val]
            precision = TP / (TP_FP)
            if precision != 0.0 and precision > 0.01:
                precision = str('%.2f' % (precision * 100))  # + '%'
            elif precision == 0.0:
                precision = '0'
            TP_FPs.append(TP_FP)
            #plt.text(x_val, y_val, str(TP_FP) + '\n' + str(precision), fontsize=7, color='black', va='center', ha='center')
    # cm = np.insert(cm, max_index, TP_FNs, 1)
    # cm = np.insert(cm, max_index-1, np.append(TP_FPs, SUM), 0)
    # plt.text(max_index, max_index, str(SUM) + '\n' + str('%.2f' % (PRECISION * 100,)) + '%', color='red',
    #          va='center', ha='center', fontsize=7)   # 绘制右下角的准确率
    #plt.text(max_index, max_index, str('%.2f' % (PRECISION * 100,)) + '%', color='red', va='center', ha='center', fontsize=12)   # 绘制右下角的准确率

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    #plt.title(title)
    #plt.colorbar()
    cb = plt.colorbar(fraction=0.045,pad=0.03)    # f:长度、
    #设置colorbar的字号
    cb.ax.tick_params(labelsize=10)

    xlocations = np.array(range(len(classes)))
    label =np.array(['空物','纸巾包','瓶子','石头','球体','易拉罐','内六角','木板','鼠标','遥控器','剪刀','万用表','卷尺','海绵','胶布']) 
    # xlocations = np.array(xlocations)
    plt.xticks(xlocations, label, rotation=45)#,labels=label)
    plt.yticks(xlocations, label, rotation=45)#,labels=label)
    # plt.ylabel('True label')
    # plt.xlabel('predict label')
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    # plt.gcf().subplots_adjust(bottom=0.15)
    # show confusion matrix
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    plt.savefig(savename, format='png',dpi=600)  # ,bbox_inches='tight'
    # savefig(fname, dpi=600, facecolor='w', edgecolor='w',
    #     orientation='portrait', papertype=None, format=None,
    #     transparent=False, bbox_inches=None, pad_inches=0.1,
    #     frameon=None, metadata=None)

    plt.show()
    # plt.show(cmap='YlOrBr')



def loadMetadata(filename, silent=False):
    '''
    Loads matlab mat file and formats it for simple use.
    '''
    try:
        if not silent:
            print('\tReading metadata from %s...' % filename)
        # metadata = sio.loadmat(filename, squeeze_me=True, struct_as_record=False)
        metadata = MatReader().loadmat(filename)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata

def preparePath(path, clear=False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path

class MatReader(object):
    '''
    Loads matlab mat file and formats it for simple use.
    '''

    def __init__(self, flatten1D=True):
        self.flatten1D = flatten1D

    def loadmat(self, filename):
        meta = sio.loadmat(filename, struct_as_record=False)

        meta.pop('__header__', None)
        meta.pop('__version__', None)
        meta.pop('__globals__', None)

        meta = self._squeezeItem(meta)
        return meta
    def _squeezeItem(self, item):
        if isinstance(item, np.ndarray):
            if item.dtype == np.object:
                if item.size == 1:
                    item = item[0, 0]
                else:
                    item = item.squeeze()
            elif item.dtype.type is np.str_:
                item = str(item.squeeze())
            elif self.flatten1D and len(item.shape) == 2 and (item.shape[0] == 1 or item.shape[1] == 1):
                # import pdb; pdb.set_trace()
                item = item.flatten()

            if isinstance(item, np.ndarray) and item.dtype == np.object:
                # import pdb; pdb.set_trace()
                # for v in np.nditer(item, flags=['refs_ok'], op_flags=['readwrite']):
                #    v[...] = self._squeezeItem(v)
                it = np.nditer(item, flags=['multi_index', 'refs_ok'], op_flags=['readwrite'])
                while not it.finished:
                    item[it.multi_index] = self._squeezeItem(item[it.multi_index])
                    it.iternext()

        if isinstance(item, dict):
            for k, v in item.items():
                item[k] = self._squeezeItem(v)
        elif isinstance(item, sio.matlab.mio5_params.mat_struct):
            for k in item._fieldnames:
                v = getattr(item, k)
                setattr(item, k, self._squeezeItem(v))

        return item

# def creat_dir(dir_path):
#     from pathlib import *
#     path =Path.cwd()
#     a =Path.joinpath(path,"c")
#     if Path(dir_path).exists():
#         print('exists')
#     else:
#         Path.mkdir(Path(dir_path))
def mdkir_test(csv_dir,csv_data_path):

    #increment_path(csv_data_path,exist_ok=True)
    while not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    while not os.path.exists(csv_data_path):
        os.mknod(csv_data_path)   # p2.0写法 3.4以上用pathlib
    # with open(csv_data_path, 'ab') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(press_imu_data)
def plt_mat(data):
    '''
    @ 矩阵热力图
    :param data:
    :return:
    '''
    #print(data)

    plt.matshow(data,cmap=plt.cm.Reds)
    plt.axis('off')
    plt.show()


def show():
    plot_confusion_matrix(cm/4000, 'cm_' + 'model' + '.png', title='confusion matrix of ' + 'model')

def debug_data():
    mat_data_dict = loadMetadata('../../data/classification/v2.mat')

    pressure = mat_data_dict['pressure']
    # pressure = np.clip(mat_data_dict['pressure'].astype(np.float32) , -1000, 5000) # np.clip torch.clamp

    pressure = pressure[:, 2, 0, 3]  # f C M U
    # plt_mat(pressure)
    plt.plot(pressure)
    plt.show()

if __name__ == "__main__":
    show()
    # debug_data()
    a= 3
    b=4
    print(id(a))
    print(id(b))





