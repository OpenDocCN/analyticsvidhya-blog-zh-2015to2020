# 在 Pytorch 中为自定义数据集(神经网络)编写数据加载器

> 原文：<https://medium.com/analytics-vidhya/writing-a-custom-dataloader-for-a-simple-neural-network-in-pytorch-a310bea680af?source=collection_archive---------0----------------------->

![](img/b6215bb1a8980a7061c991f1e796938b.png)

这个博客是为那些已经在 Pytorch 教程中看到如何使用数据加载器，并且想知道如何为数据集编写自定义数据加载器的程序员而写的。我想每个人都会同意，按照框架/算法要求的格式收集、预处理和馈送数据是一项艰巨的任务。在这篇博客中，我们将专注于创建您自己的数据集，并准备好将数据批次(数据加载器)输入 pytorch 神经网络架构。

我们将在波士顿数据集上工作。你可以在这里下载数据集[。](https://github.com/bhuvanakundumani/pytorch_Dataloader/tree/master/data)这是根据美国人口普查局收集的马萨诸塞州波士顿地区的住房信息得出的。它有 506 个观察值，13 个输入变量，1 个 ID 列和 1 个输出/目标变量。MEDV 是以千美元为单位的自有住房的中值。显然，这是一个回归问题，因为输出/目标变量是数字的(或连续的)。让我们下载程序所需的库。

```
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
```

我们要把这个转化成一个分类问题。让我们使目标变量离散或分类。我们可以在[熊猫](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html)中对数据帧的列进行桶或箱操作。我们将有两个标签为 0 和 1 的容器——介于(0–30)之间的值将为 0，介于(30–50)之间的值将为 1。

```
bins = [0,30,50]
labels = [0,1]
train_data[‘medv’] = pd.cut(train_data[‘medv’], bins=bins, labels=labels)
print (train_data[‘medv’].head())
print(train_data.head())
```

列“chas”是一个分类变量，数据集中还有一个“ID”变量。我们不希望这些出现在我们的模型中。让我们将这些列放到我们的数据框架中。除了“chas”、“ID”和“medv”之外，数字特征将具有我们的所有特征。

```
id_col = ['ID']
categorical_features = ['chas'] 
target_feature = 'medv'dropped_cols = id_col+categorical_features
train_data = train_data.drop(dropped_cols, axis=1)
all_features = train_data.columns.tolist()  #this will not have 'chas' and 'ID'numerical_features = list(set(all_features)- set([target_feature]))
```

我们需要一个训练数据集和验证数据集来训练和测试我们的网络。我们可以使用 Scikit Learn 的 train_test_split 功能来实现同样的目的。最后，Train_data 有我们的训练数据集，Valid_data 有我们的验证数据集。我们的训练数据集有 404 个观察值，验证数据集有 102 个观察值。

pytorch 中的[数据集](http://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)类和[数据加载器](http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader)类帮助我们将自己的训练数据输入网络。数据集类用于提供访问数据集中所有定型或测试样本的接口。为了实现这一点，您必须实现至少两个方法，`__getitem__`和`__len__`，以便每个训练样本都可以通过其索引来访问。在类的初始化部分，我们加载数据集(作为 float 类型)并将它们转换成 Float torch 张量。`__getitem__` 将返回特性和目标值。

```
class oversampdata(Dataset): def __init__(self, data):
        self.data = torch.FloatTensor(data.values.astype('float'))

   def __len__(self):
        return len(self.data) def __getitem__(self, index):
        target = self.data[index][-1]
        data_val = self.data[index] [:-1]
        return data_val,targettrain_dataset = oversampdata(Train_data)
valid_dataset = oversampdata(Valid_data)
```

虽然我们可以使用 Dataset 类访问所有的训练数据，但对于深度学习，我们需要批处理、洗牌、多进程数据加载等。DataLoader 类帮助我们做到这一点。DataLoader 类接受数据集和其他参数，如`batch_size`、`batch_sampler`和数量`workers`来加载数据。然后我们可以迭代`Dataloader`来获得一批训练数据并训练我们的模型。我们已经定义了 Train_Batch_Size= 101 和 Test_Batch_size =61(因为我们在训练数据集中有 404 个观察值，在验证数据集中有 102 个观察值)

```
device = "cuda" if torch.cuda.is_available() else "cpu"
kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}train_loader = DataLoader(train_dataset, batch_size=Train_Batch_Size, shuffle=True, **kwargs)
test_loader = DataLoader(valid_dataset, batch_size=Test_Batch_Size, shuffle=False, **kwargs)
```

如果 torch.cuda.is_available()，我们设置 device ="cuda "。这使得程序可以根据 GPU 的可用性在 GPU 或 CPU 上运行。在 kwargs 中，我们将 num_workers 设置为 1，将 pin_memory 设置为 True。`num_workers`表示并行生成批次的流程数量。将`num_workers`设置为正整数将开启多进程数据加载，加载器工作进程的数量为指定数量。对于数据加载，将`pin_memory=True`传递给`[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)`会自动将提取的数据张量放入固定内存，从而使数据更快地传输到支持 CUDA 的 GPU。

我们都准备好训练我们的模型了。同样的神经网络架构的细节将在下一篇[中期文章中提供。](https://link.medium.com/PoCKEbES3Z)代码可从 github-[https://github.com/bhuvanakundumani/pytorch_Dataloader](https://github.com/bhuvanakundumani/pytorch_Dataloader)获得

参考资料:

[](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) [## PyTorch 数据加载器的详细示例

### pytorch 数据加载器由 Afshine Amidi 和 Shervine Amidi 并行开发的大型数据集您是否曾经加载过这样的数据集…

stanford.edu](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)