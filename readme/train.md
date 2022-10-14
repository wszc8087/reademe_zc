# 源码细读

笔者最近训练了一个yolov5的模型，想作为预训练模型，对另外一个更为复杂的数据集进行迁移学习。但是直接使用训练得到的best.pt出现了以下问题：

1.训练epoch计数从best.pt对应的epoch开始，而不是从0开始

2.生成的result.txt记录了best.pt对应的epoch以及之前的训练结果。

3.迁移学习训练只保存last.pt，不再保存best.pt

为了解决以上问题，查阅资料得知：训练保存的pytorch文件（.pt）里面包含的不仅仅是模型与参数，还包含了其他的东西。

使用torch.load读取pytorch文件，发现.pt文件是一个字典，包含五个key：

    # epoch——当前模型对应epoch数；

    # best_fitness——模型性能加权指标，是一个值；

    # training_results——每个epoch的mAP、loss等信息；

    # model——保存的模型

    # optimizer——优化信息，一般占.pt文件一半大小。用于迁移学习的话，实测保不保留优化信息无影响
