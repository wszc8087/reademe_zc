import torch
import random
from torch import nn

random.seed(0)
torch.manual_seed(0)
input = torch.randn(3, 3)
out = torch.sigmoid(input)
target = torch.FloatTensor([[0, 1, 1],
                            [0, 0, 1],
                            [1, 0, 1]])
l1 = nn.BCELoss()
loss1 = l1(out, target) 
print(loss1)  # tensor(1.1805)

input2 = torch.FloatTensor([[1.5410, -0.2934, -2.1788],
                            [0.5684, -1.0845, -1.3986],
                            [0.4033,  0.8380, -0.7193]])
l2 = nn.BCEWithLogitsLoss()
loss2 = l2(input2, target)
print(loss2)  # tensor(1.1805)

