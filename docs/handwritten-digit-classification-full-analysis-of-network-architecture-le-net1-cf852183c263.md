# 手写数字分类网络结构全分析(Le-Net-1)

> 原文：<https://medium.com/analytics-vidhya/handwritten-digit-classification-full-analysis-of-network-architecture-le-net1-cf852183c263?source=collection_archive---------9----------------------->

![](img/994350f9cab0c087947f391c7895bffa.png)

[Pop &斑马](https://unsplash.com/@popnzebra?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

在了解手写数字分类之前，先看这个问题的应用。

**应用手写数字分类**

*   识别车辆的牌照。
*   处理手工填写的表格中的数字条目。
*   处理银行支票号码，金额，日期。
*   从纸张或图像中识别数字。

在手写数字分类问题中，图像被直接输入到学习网络中，而不是图像的固定特征向量，然后使用反向传播来学习特征向量。

为了设计一个可以概括问题的网络，我们需要充分考虑网络的架构。解决任何独特的问题或改善网络的**架构。**例如，如果现有网络过度拟合或花费更多时间进行训练，那么您可以考虑重新设计网络，使自由参数最小化。

**为什么要手写数字分类？**

*   简单的问题
*   它只包含黑色和白色像素
*   数字可以与背景分开

网络的输入是归一化图像。

在形状识别中，我们结合局部特征来检测目标。我们也可以使用局部特征来检测字符，但是这些局部特征是由学习网络决定的。

**手写数字的问题**

*   每个人都有不同的书写风格，因此手写数字的大小、宽度和方向可能因人而异。

**数字可能位于不同的位置，网络如何处理:**

*   用局部感受野扫描图像，并将状态存储在神经元中。比如在图像上滑动内核大小并执行卷积，然后执行挤压函数并存储输出。每个输出的集合被称为特征图，它代表下一层。
*   局部感受野对于每个特征图共享相同的权重。这种技术被称为权重共享，这种技术的好处是减少了自由参数的数量。
*   拥有不同的特征图将从图像中提取不同的特征，这在后续步骤中可能需要。
*   局部感受野和卷积特征映射的思想可以应用于后续的隐藏层，以提取特征。
*   在较高位置提取的特征受位置变化的影响较小。

**网络整体流量数据:**

*   第一隐藏层提取一些特征图，随后是下一隐藏层，其执行局部平均、子采样、降低特征图的分辨率，然后再次进行一些特征提取、再次平均、降低。最终输出层与最后一个隐藏层相连。

# 网络体系结构

28×28 输入图像>
四个 24×24 特征映射卷积层(5×5 大小)>
平均池层(2×2 大小)>
八个 12×12 特征映射卷积层(5×5 大小)>
平均池层(2×2 大小)>
直接完全连接到输出

# Lenet-1 的实施

**必要进口**

```
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
```

**下载训练和测试数据集。**

*   **改造。ToTensor():** 将图像转换成 Torch 张量。

```
train = datasets.MNIST("", train = True, download = True,
                      transform = transforms.Compose([
                          transforms.ToTensor()
                      ]))
test = datasets.MNIST("", train = False, download = True, 
                     transform = transforms.Compose([
                         transforms.ToTensor()
                     ]))
```

**加载数据，设置批量大小并混洗数据。**

```
trainset = torch.utils.data.DataLoader(train, batch_size = 8, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 8, shuffle = True)
```

**LeNet-1 网络:**

```
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # first layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size = 5)
        # third layer
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels=12, kernel_size=5)
        #output layer
        self.fc_out = nn.Linear(in_features=192, out_features=10)

    def forward(self, x):
        # applying activation on convolution result
        x = F.relu(self.conv1(x))
        # applying averging, Second layer
        x = F.avg_pool2d(x,kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2)
        s = self.get_dimension_size(x)
        x = self.fc_out(x.view(-1, s))

        return F.log_softmax(x, dim = 1)

    def get_dimension_size(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

**训练模型**

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)       
net = Net().to(device)
print(net)optimizer = optim.Adam(net.parameters(), lr = 0.001)
EPOCHS = 20
loss_list = []
for epoch in range(EPOCHS):
    correct = 0
    total = 0
    for data in trainset:
        X, y = data
        net.zero_grad()
        X, y = X.to(device), y.to(device)
        output = net.forward(X)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    loss_list.append(np.sum(loss.item()))
```

**定义评价函数**

```
def evaluation(dataloader):
    total, correct = 0, 0
    #keeping the network in evaluation mode 
    net.eval()
    for data in dataloader:
        inputs, labels = data
        #moving the inputs and labels to gpu
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net.forward(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        return 100 * correct / total
```

**#保存训练好的模型**

```
torch.save(net.state_dict(), 'digit_model.pth')
```

**不断学习，勇于创新。**

[](/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17) [## 综述:LeNet-1、LeNet-4、LeNet-5、Boosted LeNet-4(图像分类)

### 已经有大量的文献对 LeNet 这一经典的图像分类方法进行了深入的研究

medium.com](/@sh.tsang/paper-brief-review-of-lenet-1-lenet-4-lenet-5-boosted-lenet-4-image-classification-1f5f809dbf17)