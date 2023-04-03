# 寻找数据块天堂(fastai 数据块 API 之旅)——第 2 部分

> 原文：<https://medium.com/analytics-vidhya/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-part-2-9b23ea5d83ee?source=collection_archive---------11----------------------->

![](img/cfecba7e31c0b8972249b249ea018921.png)

本文描述了如何训练我们在本系列的[第 1 部分中构建的自定义 fast.ai 项目列表(和其他自定义数据块 API 位)。如果您还没有这样做，请确保您已经阅读了第一篇文章，并且可以运行 yelp-00 笔记本中位于](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4)[这里](https://localhost:8889/tree/medium-finding-data-block-nirvana)的配套代码。

**第二部分的代码在这里是**[](https://github.com/ohmeow/dl-experiments/tree/master/medium-finding-data-block-nirvana)****(见 yelp-01 笔记本)。****

**我在第一篇文章中所说的值得重复:**

> **我坚信*通过首先阅读和运行相关代码，然后自己手工编写代码(不要复制&粘贴),人们将会学到更多关于使用这个框架的知识。这并不意味着阅读文档，突出和强调重要的概念不重要(相信我，我做的比我应得的多)，只是为了让它在你的大脑中牢牢扎根，你必须**这样做**。所以获取代码，运行代码，热爱代码。利用它和这篇文章的内容，通过自己编写代码来展示个人的理解。***

**说到这里，让我强调一下第 2 部分代码中一些更有趣的部分。**

# **保持干燥**

**我将所有与数据块 API 相关的代码移到了`utils.py`文件中，然后从笔记本顶部导入所有代码。因为我可能想在其他地方重用这些代码，所以最好记住编程的黄金准则之一:**D**on t**R**EPE at**Y**yourself。**

# **对数据块 API 位的修复**

**当我第一次尝试训练我的模型时，我注意到张量在`mixed_tabular_pad_collate`函数中没有完全分组。我在第 1 部分的笔记本和`utils.py`文件中修复了这个问题，但是*我也在第 1 部分的文件中留下了旧的错误代码，这样你就可以查看输出应该是什么样子，不应该是什么样子。**我推荐**你用修正版或者之前的版本运行`yelp-00-custom-itemlist`笔记本，自己看看区别。***

# **微调 LM**

**由于我们正在处理文本，我认为使用数据集中可用的目标文本对基于 AWD·LSTM 的 [ULMFit 模型](https://arxiv.org/abs/1801.06146)进行微调是有意义的。参见笔记本的 **LM 微调**部分。我说明了这样做所需的基本步骤，并且我确信这是可以改进的许多地方之一。**

# **构建混合表数据中心**

**还记得我们在第 1 部分中编写所有代码吗？好吧，所有这些艰苦的工作使得在我们手头的建模任务中实际使用它变得如此简单。**

```
data_cls = (MixedTabularList.from_df(
                            train_df, cat_cols, cont_cols, txt_cols,
                            vocab=data_lm.train_ds.vocab, 
                            procs=procs, path=PATH)
          .split_by_rand_pct(valid_pct=0.1, seed=42)
          .label_from_df(dep_var)
          .databunch(bs=32))
```

**这对于任何使用 fast.ai 框架的人来说应该非常熟悉。注意我们是如何使用上面微调过的语言模型中的`vocab`。**

# **目录文本**

**我们如何使用这个[数据束](https://docs.fast.ai/basic_data.html#DataBunch)？我肯定有比我在这里提出的方法更好的方法，但是我仅仅通过利用从`tabular_learner`和`text_classifier_learner` fast.ai 学习者创建的模型就能够得到不错的结果。我绝对相信这种方法至少是新颖的(至少我还没有在任何地方看到过),并且有可能得到改进。**

**至于上面两个学习者需要的配置，我决定使用一个简单的字典来简化实验。参见在`TabularTextNN`模块定义上方声明的各自的`tabular_args`和`text_args`变量。**

**模块的`init`是所有有趣事物的所在:**

```
def __init__(self, data, tab_layers, tab_args={}, text_args={}):
        super().__init__()

        tab_learner = tabular_learner(data, tab_layers, **tab_args)
        tab_learner.model.layers = tab_learner.model.layers[:-1]
        self.tabular_model = tab_learner.model text_class_learner = text_classifier_learner(data, AWD_LSTM, 
                                                     **text_args)
        text_class_learner.load_encoder('lm_ft_enc')
        self.text_enc_model = /         
                        list(text_class_learner.model.children())[0]

        self.bn_concat = nn.BatchNorm1d(400*3+100)

        self.lin = nn.Linear(400*3+100, 50)
        self.final_lin = nn.Linear(50, data.c)
```

**如果您查看这里的表格学习器返回的模型，您会看到最后一层是一个线性层，它输出模型需要预测的预期标签数。由于我们将在获取标签的概率之前将这个模型的输出与文本输出合并，我们只需将模型层设置为等于`tabular_learner.model.layers[:-1]`就可以了。**

**类似地，我们在这里只需要文本分类学习器的编码器，因此我们通过`list(text_class_learner.model.children())[0]`将`PoolingClassifier`从其中移除。学习如何操作 PyTorch 模型对理解非常有帮助，我在下面提供了一些对我有指导意义的资源。**

**我们的`forward()`函数的最后一步是连接两个模型的结果，通过一个批量标准化层和几个线性层运行它们，以获得我们的预测值。请注意，我还使用了 fast.ai 中使用的 concat pooling 技巧来利用文本编码器返回的所有信息。**

# **培养**

**你猜怎么着？你就像训练其他 fast.ai 模型一样训练它。这意味着这里真的没有什么新东西可学。你可以把这个模型放在学习者身上:**

```
model = TabularTextNN(data_cls, tab_layers, tabular_args, text_args)
learn = Learner(data_cls, model, metrics=[accuracy])
```

**不错吧。**

# **后续步骤**

**你能打败我最好的准确度`.673`吗？**

**我很高兴接受任何和所有用你自己的笔记本提出的利用`MixedTabular`项目列表来改进我的结果的请求(它们肯定可以被改进)。也许你们中的一些人想提交一个包含一些 EDA 工作的笔记本？或者是一台笔记本电脑，它展示了一种可靠的方法，可以根据功能的重要性来决定哪些功能应该包含，哪些不应该包含？或者，也许有人可以用特征工程和/或数据扩充的方式说明一些有帮助的东西，以帮助改进我的结果？**

**无论哪种方式，**我都希望看到从这些文章中受益的社区的一些工作** …这些工作可以反过来使我自己和其他人受益。这是我对你的挑战。**

**一如既往，我希望这两篇文章对你的工作有所帮助，并随时在 twitter 上关注我，地址是 [@wgilliam](https://twitter.com/waydegilliam) 。在接下来的几个月里，我将发表一系列文章，向您展示如何使用即将发布的 fast.ai 框架第 2 版完成所有这些工作，敬请关注。**

**![](img/c3e474afa948caeb7437b45ff4945561.png)**

# **资源**

****数据块 API**
[fast . ai 文档](https://docs.fast.ai/data_block.html)
[寻找数据块涅槃—第 1 部分](https://blog.usejournal.com/finding-data-block-nirvana-a-journey-through-the-fastai-data-block-api-c38210537fe4)**

****修改 PyTorch 模型**
[如何删除预训练模型中的图层？](https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648)
[如何依次替换最后一层](https://discuss.pytorch.org/t/how-to-replace-last-layer-in-sequential/14422)
[这篇 stackoverflow 文章有一些不错的信息](https://stackoverflow.com/questions/52548174/how-to-remove-the-last-fc-layer-from-a-resnet-model-in-pytorch)
[参见官方文档中的 add_module 方法](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.add_module)**