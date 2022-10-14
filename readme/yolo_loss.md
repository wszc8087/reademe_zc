# loss.py代码解读

## 1. 标签平滑策略smooth_BCE
实际上是一种正则化策略，减少了真实样本标签的类别在计算损失函数时的权重，最终起到抑制过拟合的效果

```
def smooth_BCE(eps=0.1):
    """用在ComputeLoss类中
    标签平滑操作  [1, 0]  =>  [0.95, 0.05]
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    https://arxiv.org/pdf/1902.04103.pdf eqn 3
    :params eps: 平滑参数
    :return positive, negative label smoothing BCE targets  两个值分别代表正样本和负样本的标签取值
            原先的正样本=1 负样本=0 改为 正样本=1.0 - 0.5 * eps  负样本=0.5 * eps
    """
    return 1.0 - 0.5 * eps, 0.5 * eps
```

## 2. BCEBlurWithLogitsLoss
 YOLOV5作者实验性的函数

## 3. FocalLoss
FocalLoss损失函数来自 Kaiming He在2017年发表的一篇论文：Focal Loss for Dense Object Detection. 这篇论文设计的主要思路: 希望那些hard examples对损失的贡献变大，使网络更倾向于从这些样本上学习。防止由于easy examples过多，主导整个损失函数。

优点：
1. 解决了one-stage object detection中图片中正负样本（前景和背景）不均衡的问题；
2. 降低简单样本的权重，使损失函数更关注困难样本；

## 4. QFocalLoss

## 5. ComputeLoss类
### 5.1 init函数
### 5.2 build_targets

这个函数是用来为所有GT筛选相应的anchor正样本。筛选条件是比较GT和anchor的宽比和高比，大于一定的阈值就是负样本，反之正样本。筛选到的正样本信息（image_index, anchor_index, gridy, gridx），传入__call__函数，通过这个信息去筛选pred每个grid预测得到的信息，保留对应grid_cell上的正样本。通过build_targets筛选的GT中的正样本和pred筛选出的对应位置的预测样本进行计算损失。

补充理解：这个函数的目的是为了每个gt匹配相应的高质量anchor正样本参与损失计算，j = torch.max(r, 1. / r).max(2)[0] < self.hyp[‘anchor_t’]这步的比较是为了将gt分配到不同层上去检测（和你说的差不多），后面的步骤是为了将确定在这层检测的gt中心坐标，进而确定这个gt在这层哪个grid cell进行检测。做到这一步也就做到了为每个gt匹配anchor正样本的目的。



### torch语法
PyTorch中的repeat()函数可以对张量进行重复扩充。
当参数只有两个时：（h的重复倍数，w的重复倍数）。1表示不重复
当参数有三个时：（通道数的重复倍数，h的重复倍数，w的重复倍数）。
```
a= torch.arange(30).reshape(5,6)
print(a)
print('b:',a.repeat(2,1).shape)   # 10 12
print('c:',a.repeat(2,1,1)) # 2 5 6
```