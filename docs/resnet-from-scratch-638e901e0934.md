# 雷斯内特:从零开始

> 原文：<https://medium.com/analytics-vidhya/resnet-from-scratch-638e901e0934?source=collection_archive---------17----------------------->

CNN 架构是分析图像和视频图形材料的一些重要形式。同样的一些重要应用也可以在生物医学工程系看到，特别是在生物成像领域。根据文献综述，已经设计出了许多这样的 CNN 架构，并且已经实现来研究人体中异常的检测。在这里，我将探索“ResNet 的制作:从零开始”

> 模块:PyTorch、Cuda(可选)

如果你对如何在你的系统中安装 PyTorch 感到困惑，那么你可能想看看这里的链接。它会帮助你！向前发展…

# **雷斯网:**

这是一个有很多更深层次的深层架构。深层网络用于更好地理解可用图像的重要方面和特征。让我们建立…

```
import torch
import torch.nn as nn
```

这是您将导入到您的环境中的重要的两行。现在，**让我警告你！**这是一个庞大的网络，需要一些时间来完成数据集的工作。对于新手来说，如果你看到你的环境屏幕长时间停留在一个位置，不要惊慌，因为计算机只是在做你要求它做的事情！

一点编码经验是可以预期的，因为我们将要定义类，这最终会涉及到一些继承问题，这可能会让你头脑发晕。因此，为了阻止这种情况发生，请投入一些关于 OOPs 的[教程。](https://www.youtube.com/watch?v=SiBw7os-_zI&ab_channel=freeCodeCamp.org)

```
class resnet_block(nn.Module):
    def __init__(self, in_channels, out_channels, 
idt_downsample=None, stride=1):
        super(resnet_block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = idt_downsample
```

该代码已被构造为重复使用和最小的努力记住的东西。在初始化阶段，创建了一个 resnet 块，其中包含本文中提到的所有必要参数。由于 ResNet 工作在剩余网络理论上，因此我们需要一个额外的相同的矩阵。

现在，既然我们已经创建了我们的 resnet 块，是时候让它“前进”并定义“前进”函数了。

```
def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
```

“前进”功能被定义为在初始化阶段之后工作，并且包含所有必要的步骤(如研究论文中所定义的)。现在，无论何时调用 resnet 块，都是由“forward”函数来完成类的主要任务。现在，既然已经编写了“forward”函数，我们需要继续进行实际的 ResNet 建模。

对于 ResNet 建模，我们将遵循论文中定义的实际结构。现在，在这种情况下，我们会发现继承概念在起作用。我们将获取原始类的实例，并在当前类中实现它。如果你很好地理解了[论文](https://arxiv.org/abs/1512.03385)，那么它对你来说将是完全合乎逻辑的，你将立刻跟进！

现在，公式化“ResNet”块。

```
class ResNet(nn.Module):
    def __init__(self, resnet_block, layers, img_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers
        self.layer1 = self._layers(resnet_block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._layers(resnet_block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._layers(resnet_block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._layers(resnet_block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
```

从上面的块中，您可以观察到有另一个方法叫做' _layers '。我们还没有定义它。这将是我们模型的原始定义块。“_layers”块将具有身份下采样部分的定义(在研究论文中提到)。

现在，实际运行我们算法的 ResNet 类的主要部分是“forward”定义函数。该函数是 ResNet 架构实现的基础，每次我们调用该函数时，我们的模型都会找到实现的“转发”定义。

```
def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
```

现在，对于神秘的“层”功能。我猜，这是很多人的猜测。这是我们将堆叠层的地方，并且基于给定的输入，它将建模期望的架构版本(例如:ResNet50、ResNet101 等)。

```
def _layers(self, resnet_block, no_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                          stride=stride),
                                                nn.BatchNorm2d(out_channels*4))

        layers.append(resnet_block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(no_residual_blocks - 1):
            layers.append(resnet_block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
```

所以，我们已经定义了一切，是时候来决定我们的推测是否正确了。我们可以实现下面给出的代码块来测试架构。

```
def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(resnet_block, [3,4,6,3], img_channels, num_classes)
```

我们完了。那是一项艰巨的工作。谢谢你坚持到最后。这是 CNN 领域中的一个重要架构，理解它确实很困难！如果您需要进一步的帮助，请参阅以下部分…

# 路上帮忙！

如果你仍然感到困惑，想全面了解实际的方法和应用的情况，那么请直接点击[这个链接](https://github.com/tanmaydn/CNNfromScratch/blob/main/ResNet.py)。

# 参考

整个片段的灵感来自 ResNet 研究人员的原创作品。[敬请过目](https://arxiv.org/abs/1512.03385)！