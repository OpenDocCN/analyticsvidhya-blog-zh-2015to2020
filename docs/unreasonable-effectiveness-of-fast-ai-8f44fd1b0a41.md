# fast.ai 不合理的有效性

> 原文：<https://medium.com/analytics-vidhya/unreasonable-effectiveness-of-fast-ai-8f44fd1b0a41?source=collection_archive---------21----------------------->

![](img/38e1cb3cd01af5b4dfcce58eb8284e43.png)

塞萨尔·卡利瓦里诺·阿拉贡在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

你和我一样吗？开始非常热衷于实现一个想法或模型，然后陷入了与 [PyTorch](https://pytorch.org/) 、 [Tensorflow](https://www.tensorflow.org/) 或 [Keras](https://keras.io/) 的库问题。混淆了哪些损失流过，你计算梯度的目的是什么？不要担心，因为 [fast.ai](https://www.fast.ai/) 会保护你。

很多天来，我从来没有真正想要深入研究 fast.ai 的文档，我总是低估了库只是一个工具，它涵盖了大部分细节，不支持“真正的研究”。我一直认为，对于初学者来说，用 4-5 行构建 SOTA 模型是一种看起来很酷的简单方法。

昨天，我听了莱克斯·弗里德曼和杰瑞米·霍华德的播客。我可以理解他的许多想法，然后考虑再给它一次机会。当我今天尝试使用 fast.ai 实现一个简单的二进制分类模型时，我所有的假设都改变了。

我使用了[脊柱数据集](https://www.kaggle.com/sammy123/lower-back-pain-symptoms-dataset)，这是一个不错的数据集，需要简单的预处理和基本的网络建模。我的首选库一直是[py torch](https://pytorch.org/)；动态计算图非常灵活和直观。我必须编写大量样板代码来读取数据并成批加载，然后编写自己的训练循环，fast.ai 通过最佳实践自动完成所有这些部分，例如[一个周期训练](https://docs.fast.ai/callbacks.one_cycle.html#The-1cycle-policy)并使用 [find_lr](https://docs.fast.ai/basic_train.html#lr_find) 找到最佳学习率。在这篇博客中，我将带你了解开发过程和结果

# 安装

fastai 库依赖于 PyTorch 库，需要安装一个工作的 Torch。最好只是创造一个新的康达环境，以避免任何冲突。

```
conda create --name fastai
conda activate fastai
conda install -c pytorch -c fastai fastai
```

它们还提供了一种使用 **pip 进行安装的方法。**要了解详情，关于这个你可以参考[这里](https://github.com/asvskartheek/Spine-Dataset#setting-up)。

# 资料组

你可以在这里找到数据集，我希望我能提供一个 shell 脚本来下载它，但是 [Kaggle](https://www.kaggle.com/) 没有提供一个简洁的方法来做到这一点。解压缩下载的 zip 文件，您将获得一个数据集文件。

# 预处理

```
df = pd.read_csv('Dataset_spine.csv')
df.drop('Unnamed: 13',axis=1,inplace=True) # Drop useless featdf['Class_att'] = df['Class_att'].astype('category')
encode_map = {  'Abnormal': 1,  'Normal': 0}
df['Class_att'].replace(encode_map, inplace=True)  # 1 HotEncodingX = np.array(df.iloc[:, 0:-1]).astype(np.float64) # Features
y = np.array(df.iloc[:, -1]).astype(np.float64) # Labels
```

# 模型

```
class BinaryClassification(nn.Module):    
  def __init__(self,input_feats,hidden_size):
    super(BinaryClassification, self).__init__()
    self.layer_1 = nn.Linear(input_feats, hidden_size)
    self.layer_out = nn.Linear(hidden_size, 1)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()def forward(self, inputs):
    x = self.relu(self.layer_1(inputs))
    x = self.layer_out(x)
    x = self.sigmoid(x)
    return x
```

# 训练循环

训练循环是最好的部分，不再需要 model.backward()、model(x)和数十行代码来训练模型，不再需要样板代码。(看着你，PyTorch..)

```
learner.fit_one_cycle(n_epochs, lr)
learner.save('model')
```

看看代码有多可爱就知道了。我太喜欢 fast.ai 了

## 骗局

不要对 fast.ai 撒谎，它也不是玫瑰人生，文档也不是很清楚，尤其是对于 import 语句。我不得不挣扎一下，就一点点。

# 食品新闻

好消息，你不用再做了。我已经完成并发布了框架库，您可以在未来的项目中使用它。它对您使用的体系结构和数据集具有响应性和灵活性。我将发布整个代码库，里面有清晰的说明。

你所要做的只是分叉它，并在 model.py 文件中更改数据集和模型架构。事不宜迟，下面是到存储库的[链接。如果你喜欢，那就开始吧，关注我。README 有所有的说明，如果你有任何意见和想法，打开一个问题，我很乐意讨论或在下面留下评论。](https://github.com/asvskartheek/Training-Framework)