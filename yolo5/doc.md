# 损失函数说明


## 损失函数

1. pytorch中nn.CrossEntropyLoss()：交叉熵损失函数，用于解决多分类问题，也可用于解决二分类问题。 在使用nn.CrossEntropyLoss()时其内部会自动加上Sofrmax层。
nn.CrossEntropyLoss(x,y) = nn.Logsoftmax() + nn.NLLLoss()

2. BCELoss是Binary CrossEntropyLoss的缩写，nn.BCELoss()为二分类交叉熵损失函数，只能解决二分类问题。 在使用nn.BCELoss()作为损失函数时，需要在该层前面加上Sigmoid函数，一般使用nn.Sigmoid()即可

nn.BCEWithLogitsLoss() = nn.sigmoid() + nn.BCELoss()

BCELoss(x,y)和BCEWithLogitsLoss(x,y)的 x 和目标 y：label（必须是one_hot形式）的形状相同。如下代码；
CrossEntropyLoss(x,y）：如果是多分类，x: 预测值，是上一层网络的输出，size=[batch_size, class]，如网络的batch size为120，数据分为10类，则x 的size=[120, 10]；y：target是数据的真实标签，是标量，size=[batch_size，1]，如网络的batch size为120，则size=[120，1]，值为0~num_classes-1之间。如下代码；
