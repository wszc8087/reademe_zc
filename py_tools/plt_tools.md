# Demo1-混淆矩阵

@[TOC](xxxx)

# 使用库
from sklearn.metrics import confusion_matrix


1. demo1
```
# ----------外框设置------------
    # plt.axis('off')  # 一次性去掉所有外框
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)  # 选择性去掉某个轴
    ax.spines['top'].set_visible(False)  #

    # ---------plt坐标轴中文------------
    plt.xlabel("采样点数据/帧",fontsize=14)
    plt.ylabel("法向磁通量变化/mT",fontsize=14)
    plt.rcParams['font.sans-serif'] ='SimHei'
    plt.rcParams['axes.unicode_minus'] = False

        my_x_ticks = np.arange(0, 1750, 200)  # 原始数据有8个点，故此处为设置从0开始，间隔为1
    my_y_ticks = np.arange(0, 3, 0.2)
    plt.xticks(my_x_ticks,fontsize=14)
    plt.yticks(my_y_ticks,fontsize=14)
    plt.xlim(0, 1760)
    plt.ylim(-0.1, 2.4)

    # ---图例说明---
    plt.legend([p0, p1, p2,p3, p4, p5,p6],
               labels=['0.2N', '0.4N', '0.6N','0.8N','1.0N', '1.2N','1.4N'], 
               loc='best',
               frameon=False,  # 图例边框是否需要
               fontsize=10
               )
    save_name ='test_1_plt_show.tif'
    plt.savefig(save_name,dpi=600,bbox_inches='tight')  # ,bbox_inches='tight'
```

## a