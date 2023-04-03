# 用 AllenNLP 微调 BERT 的简单例子

> 原文：<https://medium.com/analytics-vidhya/fine-tuning-bert-with-allennlp-7459119b736c?source=collection_archive---------2----------------------->

## 以 SNLI 数据集为例，简单演示如何在 AllenNLP 库中微调 BERT。

![](img/a790aa53d8d7e332a1b73317238e336e.png)

芝麻街的伯特的强制性形象

听说了这么多关于[allenlp](https://allennlp.org/)图书馆的事情后，我终于腾出一些时间来熟悉它。因为我的大部分工作(最近)涉及微调预先训练的语言模型，所以我决定学习这个库，尝试用它来微调一个 BERT 模型。

有些随意地，我选择使用[斯坦福自然语言推理(SNLI)数据集](https://nlp.stanford.edu/projects/snli/)来微调 [**自然语言推理**](https://paperswithcode.com/task/natural-language-inference) 的任务。然而，我注意到用于分类的内置 BERT 模型([allennlp . models . BERT _ for _ classification](https://allenai.github.io/allennlp-docs/api/allennlp.models.bert_for_classification.html))不能直接与用于 SNLI 的内置数据集读取器([allennlp . data . dataset _ readers . SNLI](https://allenai.github.io/allennlp-docs/api/allennlp.data.dataset_readers.snli.html))一起工作。这提供了一个完美的学习机会，我决定在这篇短文中解释一下。

> 请注意，本教程并不是对 AllenNLP 的介绍。为此，我推荐 AllenNLP 官方演示[这里](https://allennlp.org/tutorials)。

# ✅入门

本教程唯一的依赖项是 AllenNLP，它可以和 pip 一起安装。首先确保你有一个干净的 Python 3.6 或者 3.7 虚拟环境，然后用 pip 安装。举个例子，

```
# Assuming conda is installed
conda create -n bert_snli python=3.7 -y
source activate bert_snlipip install allennlp
```

如果你计划在本地训练模型**，你会想要一个[支持 CUDA](https://developer.nvidia.com/cuda-zone) 的 GPU。**

最后，你需要从[这里](https://nlp.stanford.edu/projects/snli/)下载 SNLI 数据集。请记住您保存此的位置，因为我们稍后将需要此路径。

## 使用 Colab

如果你愿意的话，我也可以在 **Google Colab** 中找到这个教程。你可以跟着它走，不需要在本地安装任何东西。

# 🔧进行必要的更改

为了在 SNLI 数据集上训练 BERT，我们唯一需要修改的是 [AllenNLP SNLI 数据集读取器](https://allenai.github.io/allennlp-docs/api/allennlp.data.dataset_readers.snli.html)。

这是因为，对于像 NLI 这样的任务，输入示例是成对的序列，BERT 期望下面的输入结构(借用来自遥远星系的[著名台词](https://www.techly.com.au/2016/03/31/darth-vader-never-said-luke-i-am-your-father-in-star-wars/)作为示例):

```
[CLS] I am your father . [SEP] No . No ! That ’ s not true ! [SEP]
```

具体来说，我们需要用特殊的[SEP]标记连接我们的两个句子(在本例中是我们的*前提*和我们的*假设*)。为此，我们将以下代码添加到 AllenNLP SnliReader 中(就在这里的[周围](https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/snli.py#L76-L78)):

```
*# Join the premise and the hypothesis.
# Because we will use* bert-pretrained as token indexer, 
# both the premise and hypothesis already contain 
# [CLS] and [SEP] tokens. We simply remove the [CLS]
# token from sequence 2 (hypothesis) and concatenate it with
# sequence 1 (premise)tokens = premise_tokens + hypothesis_tokens[1:]
fields["tokens"] = TextField(tokens, self._token_indexers)
```

在这个[要点](https://gist.github.com/JohnGiorgi/6930320f36f21cce501514a689fbb907)中为你做了什么:

> Medium 渲染 Gist 中的所有文件，这里我们指的是第一个渲染的文件(bert_snli.py)

用于 BERT 的改进的 SnliReader

# ⚙️正在写配置

最后但同样重要的是，我们需要编写一个 [jsonnet](https://jsonnet.org/) config 来使用我们的定制数据集加载器，定义我们的模型并设置训练。事实证明这非常简单:

> Medium 渲染 Gist 中的所有文件，这里，我们指的是第二个渲染文件(train.jsonnet)

用于定义模型、数据集读取器和训练制度的配置文件

注意，我们将自己的子类 BertSnliReader 作为“type”提供给“dataset_reader”。

# 🚀训练模型

最后，我们准备训练模型。首先下载 train.jsonnet 和 bert_snli.py 文件(见[要诀](https://gist.github.com/JohnGiorgi/6930320f36f21cce501514a689fbb907))。然后，将“train_data_path”和“validation_data_path”更改为 SNLI 数据集的本地副本的路径。

激活虚拟环境后，我们可以启动如下培训课程:

```
allennlp train train.jsonnet --include-package bert_snli -s ./tmp -f
```

> 请注意，这将把序列化模型保存在。/tmp，如有必要，覆盖此目录。

加载数据集后，您应该会看到训练进度输出，例如:

```
accuracy: 0.7972, loss: 0.5115 ||:  10%|#         | 1761/17168 [18:18<2:31:52,  1.69it/s]
```

# ♻️总结

就是这样！用 AllenNLP 微调 BERT 轻而易举。对于以句子对作为输入的任务，我们只需要修改现有的数据集读取器，用 BERTs 特殊的[SEP]标记连接句子。

有很多事情可以改善这个设置，比如

*   从一个 URL 下载 SNLI 数据集，并将其缓存在磁盘上(我相信这可以用 AllenNLP 实现，但我无法弄清楚)。
*   添加一个学习率调度器，类似于在[原始 BERT 实现](https://www.aclweb.org/anthology/N19-1423/)中使用的那个。
*   编写一个[预测器](https://allenai.github.io/allennlp-docs/api/allennlp.predictors.html)，并提供一个经过训练的模型作为一个简单的网络服务。

如果你对本教程有任何疑问，请在这里留下你的评论🙌