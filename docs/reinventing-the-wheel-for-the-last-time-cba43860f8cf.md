# 最后一次重新发明轮子

> 原文：<https://medium.com/analytics-vidhya/reinventing-the-wheel-for-the-last-time-cba43860f8cf?source=collection_archive---------17----------------------->

# **TL；DR；**

不要浪费时间多此一举。如果你只是拥抱一个专注于 ML/DL 软件基础设施和管道的开源项目，你的 DL 代码(用于成像和其他)会更有效。

其中一个项目是 eisen . ai([http://eisen . ai](http://eisen.ai))，它是完全模块化的，允许你只用几行代码就建立一个完整的训练、验证和测试管道。它将为您执行模型并行、自动混合精度和数据并行，并以完全自动的方式将摘要导出到 tensorboard 等。文档在 [http://docs.eisen.ai](http://docs.eisen.ai)

我坚信，人工智能研究人员和开发人员不应该每次开始新的东西时都必须从头开始编码。同时，我也相信开发应该在跨团队共享的代码基础上进行，并且为大量开发人员所熟知，甚至跨组织。

如果你还在编码，那你就做错了。

```
def train(model, loss, optimiser, dataset, epochs):
    for epoch in range(epochs):
        for input, label in dataset:
            output = model(input)
            loss = loss(label, output)
            loss.backward()
            optimiser.step()
            # ...
```

而你做错的原因是，你只是在解决一个对你来说特定的问题，而且只会存在有限的时间，然后稍微改变一下，让你所有的工作变得毫无用处。你会不可避免地发现自己一遍又一遍地编写这个训练循环，因为你太懒了，以一种可以适应任何网络、任何任务和任何数量的损失以及你想要计算的额外结果的方式来编写它。此外，因为没有人是完美的，你可能会在这里或那里有小的不完美，这将花费你几天的额外工作。

即使您正在使用 keras，并且您认为您做的一切都是正确的，因为在数百行数据加载、操作和转换之后，您可以轻松地编写:

```
model.fit(data, labels, epochs=10, batch_size=32)
```

您仍然是错误的，因为如果不进行几乎完全的重写，您的代码很可能不够模块化和灵活，无法经受开发过程中需求、不同数据源和格式的变化。你写的那几百行代码就是问题所在！或者更好地说，你写这几行的方式就是问题所在。

让我们不要谈论新团队成员的入职！你可能需要花上几周的时间来解释 main.py 文件中一个看起来很小很简单的 if-then-else 语句是如何导致程序在 3 个不同模块的 5 个函数中执行的。另一方面，你有 2 个月的时间来完成一个需要 6 个月的项目，你没有时间产生结果，更不用说适当地记录你的代码了。

也许你有一个巨大的功能叫做预处理，它为你做“一切”:

```
def pre_processing(data_path):
    labels_df = pd.read_csv(os.path.join(data_path, "filename.csv"))
    png = [f for f in os.listdir(data_path) if 'png' in f]
    for png_name in png:   
        im = Image.open(os.path.join(data_path, png_name))
        labels_row = labels_df.loc[png_name]

        if labels_row['type'] == 'colour':
            im = pre_processing_colour(im)
            if labels_row['city'] == 'New York':
                 im = pre_processing_nyc(im)
            else:
                 im = crop(im, [224, 224])
                 im = grayscale(im) # ...
     # ...
```

唯一的例外是，你只是在搬起石头砸自己的脚，因为它太复杂、太难阅读，任何人都无法重用(深度学习是一个持续的过程，只要它解决的问题存在，它就会持续下去)。

你花了太多时间编码。即使编码是好的，因为它让我们有成就感，并且是告诉老板你很忙的一个很好的理由，它也不能永远继续下去。我们需要做一些有意义的事情，感觉自我实现了。

![](img/ebbd541f68b48688fde6cd3f69bd5a22.png)

从谁制作了谁的音乐录影带“现代人的陷阱”中看到的人类需求的修正金字塔。在 https://youtu.be/ucQ3lc4lkhY 的 youtube 上观看

如果你遵循一个可靠且固执己见的设计，它被证明是灵活的，能够适应任何类型的问题、任何数量的输入和输出、任何损失、任何模型、任何指标，并且能够在一行中进行模型并行处理以及将数据自动导出到 tensorboard，我打赌你会更高兴。你知道那种早上醒来就知道自己要做什么的感觉吗？你终于可以得到那个了！

想想看…你可以醒来，而不是胡扯你自己和你的老板，你可以说:“今天我要建立一个转换，将我们的 RGB 颜色空间图像转换到实验室”。或者你可以说:“今天我将尝试在 8 个 GPU 上运行我们的模型，因此它将通过更大的批量进行扩展和学习”。

你知道你会怎么做吗？只写你需要的东西:

```
from skimage import colorclass LABTransform:
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data):
        for key in self.fields:
            data[key] = color.rgb2lab(data[key])

        return data
```

看起来我已经编写了太多的代码来实现这种转换，但是如果我告诉你，现在你可以将这个对象与任何数量和种类的其他转换混合和匹配，以便最终为你的数据实现一个复杂、模块化和强大的转换链，会怎么样？

而模型并行性呢？拿着你的模型，做:

```
from torchvision.models import resnet18 # that could be your model!from eisen.utils import EisenAutoModelParallelModuleWrappermodel = EisenAutoModelParallelModuleWrapper(
    number_gpus=8,
    split_size=16,   
    module=resnet18, 
    input_names=['image'], 
    output_names=['logits'], 
    num_classes=1000
)
```

我建议你在增加批量时不要忘记调整你的学习率，因为你能够在一行中使用流水线执行进行模型并行化([https://py torch . org/tutorials/intermediate/model _ parallel _ tutorial . html # speed-up-by-pipelined-inputs](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html#speed-up-by-pipelining-inputs))。

哦，那冲浪板呢？也许你已经厌倦了写语句把你的数据一块一块的拿到 tensorboard 上？也许你不想每次都在你的训练循环中写这个:

```
from torch.utils.tensorboard.writer import SummaryWriterwriter = SummaryWriter(log_dir=artifacts_dir)writer.add_scalar(
    'loss', 
    np.mean(loss),
    global_step=epoch)writer.add_images(
    'input', 
     input, 
     global_step=epoch, 
     dataformats='NCHW'
)# ...
```

如果您可以在一行中自动完成所有输入、输出、指标和损失，会怎么样？

```
from eisen.utils.logging import TensorboardSummaryHooktraining_tb_hook = TensorboardSummaryHook(
    training_workflow.id, 
    'Training', 
    './artifacts_one'
)
```

我给了你几个很好的理由来标准化你的代码，遵循标准的架构，并在你开发深度学习方法时最小化出错(和浪费时间)的机会。事实上有很多原因。随着 DL 成为一个行业，其流程需要能够扩展。

停止编写没有测试或文档的庞大函数，拥抱与你正在使用的环境相适应的包(例如 PyTorch ),简化并改善你的生活。我喜欢只用几行代码就能在具有挑战性的医学数据集上进行训练，我不需要每次都从头开始重写，假装我在做一些工作！看看这个例子:[http://bit.ly/2HjLlfh](http://bit.ly/2HjLlfh)在我看来很简单！

有很多解决方案。很多不同的项目活跃在深度学习的任何一个分支。其中一个项目涉及的内容与我的工作很接近:医学图像分析和计算机视觉。它的架构是完全模块化的，你已经在这篇文章中看到了它的使用的小例子。你可以在 [http://eisen.ai](http://eisen.ai) 上找到更多信息，并随意使用代码，因为它是在麻省理工学院许可下发布的。文档在 [http://docs.eisen.ai](http://docs.eisen.ai) 。

有了 Eisen，你可以使用你需要的模块，并为特定的需求编写新的代码。这里显示了主要的构建模块

![](img/cd8dc17a5e12b8eed5f160b61875fdf4.png)

艾森( [http://docs.eisen.ai](http://docs.eisen.ai) )主要组件分解

你可以用它们来定义培训、验证和测试的工作流程。

![](img/043b9d1d9b4f18fab2f2349f6f126c0c.png)

连接一堆转换，添加一个模型，一个 PyTorch 的优化器，一个数据集，损失，如果你想要一些指标，你可以运行你的算法。任何算法。任何任务。

我建议开始关注这些项目，并开始专注于构建模块化和优雅的代码来解决实际问题，而不是构建容易出错、高度个性化和不可重用的可怕的 python 代码。