# 目标检测-YOLO系列导航
网址链接: [重要参考资料](https://blog.csdn.net/qq_38253797/article/details/119763327
)
- [目标检测-YOLO系列导航](#目标检测-yolo系列导航)
  - [一、data文件夹](#一data文件夹)
    - [超参数文件](#超参数文件)
    - [数据集配置文件](#数据集配置文件)
  - [二、models文件夹](#二models文件夹)
    - [激活函数activations.py](#激活函数activationspy)
    - [常见的网络模块common.py](#常见的网络模块commonpy)
    - [实验模块experimental.py](#实验模块experimentalpy)
    - [模型搭建模块yolo.py](#模型搭建模块yolopy)
    - [转换模块export.py](#转换模块exportpy)
    - [网络配置文件yolov5s.yaml](#网络配置文件yolov5syaml)
  - [三、utils文件夹](#三utils文件夹)
    - [autoanchor.py](#autoanchorpy)
    - [数据集模块datasets.py](#数据集模块datasetspy)
    - [常用工具模块general.py](#常用工具模块generalpy)
    - [谷歌工具模块google_utils.py](#谷歌工具模块google_utilspy)
    - [损失函数模块loss.py](#损失函数模块losspy)
    - [评估计算模块metrics.py](#评估计算模块metricspy)
    - [画图模块plots.py](#画图模块plotspy)
    - [torch工具模块torch_utils.py](#torch工具模块torch_utilspy)
  - [四、训练模块train.py](#四训练模块trainpy)
  - [五、验证测试模块 val(test).py](#五验证测试模块-valtestpy)
  - [六、detect.py检测模块](#六detectpy检测模块)
  - [笔记必写模块](#笔记必写模块)

## 一、data文件夹
### 超参数文件
1. 脚本存放一些超参数的设置
* 依次包括这四部分：训练、损失函数、其他、数据增强
### 数据集配置文件
1. 这个文件是数据配置文件，存放着数据集源路径root、训练集、验证集、测试集地址，类别个数，类别名，下载地址等信息。
   
## 二、models文件夹
### 激活函数activations.py
### 常见的网络模块common.py
### 实验模块experimental.py
### 模型搭建模块yolo.py
重点掌握：包括模型的构建，顺便熟悉各基本网络
### 转换模块export.py
1. 这个部分是模型的转换部分，将模型转换为torchscript、 onnx、coreml等格式，用于后面的应用中，方便将模型加载到各种设备上。

### 网络配置文件yolov5s.yaml

## 三、utils文件夹
### autoanchor.py
1. 这个文件是通过 k-means 聚类 + 遗传算法来生成和当前数据集匹配度更高的anchors。
2. 注意点：train.py的parse_opt下的参数noautoanchor必须为False;
3. hyp.scratch.yaml下的anchors参数必须注释掉
   
### 数据集模块datasets.py

### 常用工具模块general.py
这个文件是yolov5的通用工具类，写了一些通用的工具函数，用的很广，整个项目哪里都可能用到。这个文件的函数非常多，代码量也很大（上千行了），也都比较重要。


### 谷歌工具模块google_utils.py
主要是负责从github/googleleaps/google drive 等网站或者云服务器上下载所需的一些文件。是一个工具类

### 损失函数模块loss.py
两种常用的交叉熵损失函数BCELoss和BCEWithLogitsLoss 。另外，这个文件涉及到了损失函数的计算、正负样本取样、平滑标签增强、Focalloss、QFocalloss等操作，都是比较常用的trick，一样都要弄懂！

### 评估计算模块metrics.py

这个文件存放的是计算mAP、混淆矩阵、IOU相关的函数，在看之前需要大家了解基本的目标检测指标，mAP的定义，计算方式等知识。相对来说这个文件的代码难度还是相对挺高的，需要不断的debug，debug，debug!

### 画图模块plots.py
都是一些画图函数，是一个工具类。代码本身逻辑并不难，主要是一些包的函数可能大家没见过。这里我总结了一些画图包的一些常见的画图函数： 【Opencv、ImageDraw、Matplotlib、Pandas、Seaborn】一些常见的画图函数

### torch工具模块torch_utils.py

## 四、训练模块train.py 

## 五、验证测试模块 val(test).py

## 六、detect.py检测模块


## 笔记必写模块
1.