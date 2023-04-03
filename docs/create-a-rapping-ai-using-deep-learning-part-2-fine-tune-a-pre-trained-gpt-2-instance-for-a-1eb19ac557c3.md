# 使用深度学习创建说唱人工智能

> 原文：<https://medium.com/analytics-vidhya/create-a-rapping-ai-using-deep-learning-part-2-fine-tune-a-pre-trained-gpt-2-instance-for-a-1eb19ac557c3?source=collection_archive---------7----------------------->

## 第 2 部分:为特定任务微调预先训练好的 GPT 新协议实例

问候所有**数据旅行者**和 **ML 超级英雄**！

欢迎来到本系列的**第二部分**，在这里我试图构建一个人工智能，它可以以著名说唱歌手的风格提出全新的说唱歌词，并将其转换成音轨！

![](img/29b0a23b7c40841838a9766cf6470153.png)

确保你已经阅读了这个故事的第 1 部分，在那里我们学习了如何使用并发 Python 有效地收集一个相当大的说唱歌词数据集。

本周，我们将训练一个模型来为我们生成新的说唱歌词，为此我们将使用深度学习。

自然语言的生成模型在过去几年中取得了非常快的进展。正如我在第 1 部分中警告您的，这将是一个非常实用的教程。因此，我不会在这里深究 NLP 模型的理论。但是如果你想详细了解这些模型是如何工作的(我认为你应该这样做)，我强烈推荐你阅读下面的**易于理解的博客文章**:

[循环神经网络图解指南](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)

[探索 lstm](http://blog.echen.me/2017/05/30/exploring-lstms/)

[变压器说明——你所需要的只是注意力](https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)

对于我们的说唱 AI，我们将使用一个预训练版本的 **GPT-2** 模型，该模型基于[变形金刚](https://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/)，由 OpenAI 于 2019 年 2 月发布。该模型因未向公众发布而迅速成名……**因为它太好了**！可以在这里阅读论文原文：<https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>

然而，一个更小的 3 . 45 亿参数版本的模型公开发布了。后来他们发布了一个更大的，7.74 亿参数的版本。在这个项目中，我们将加载预训练的 345M GPT-2 模型，并使用我们在[第 1 部分](/@max.y.leander/create-a-rapping-ai-using-deep-learning-part-1-collecting-the-data-634bbfa51ff5)中编译的说唱歌词数据集对其进行微调。这可以在 [Google Colab](https://colab.research.google.com/) 平台上完成，以便免费利用 GPU 培训！(不幸的是，在撰写本文时，774M 型号对 Google Colab 来说太大了，但如果你能接触到更强大的环境，一定要尝试一下。)

通俗地说，你可以说 GPT-2 有长期和短期记忆，就像大脑一样。(我知道，GPT-2 是基于[注意力](https://arxiv.org/abs/1706.03762)，而不是 [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf) ，但我认为这个类比仍然成立。)GPT-2 以 emi 监督的方式训练，这意味着使用了监督的学习和非监督的学习。

它首先以一种无人监督的方式训练了很长时间，基于非常大量的数据。这样做的目的是学习不同术语之间的关系，以及它们以何种方式对不同的上下文做出贡献。这些关系构成了语法、结构、文体等低级语言概念，代表了大脑类比中的长时记忆。

通过**微调**，GPT-2 将学习当前对我们重要的上下文和主题，同时保持其长期记忆，即其语法、结构、风格等感觉。它从无人监督的训练中学到了什么。微调是通过以监督的方式训练模型来完成的。也就是说，我们将交给它特定的输入/输出对，然后让模型在给定新输入的情况下预测输出。在我们的例子中，这个新的输入可能是一个最喜欢的说唱歌手的名字，希望这个模型能够以那个说唱歌手的风格吐出可信的歌词！

你应该做的第一件事是前往 [Google Colab](https://colab.research.google.com/) 并创建一个具有 **GPU 支持**的新笔记本。Github 用户`nshepperd`已经创建了一个很好的回购协议，可以很容易地对 GPT-2 模型进行实验，所以我们将通过在笔记本中运行以下内容，将其克隆到我们的 Google Colab 工作区中:

```
!git clone [https://github.com/nshepperd/gpt-2.git](https://github.com/nshepperd/gpt-2.git)
```

接下来，我们需要安装所有必需的 Python 包:

```
cd gpt-2
!pip3 install -r requirements.txt
```

为了访问来自 Colab 的数据集，我已经将带有说唱歌词的文本文件复制到我的 **Google Drive** 中。通过使用以下命令将它安装在 Colab 环境中，可以很容易地访问它:

```
from google.colab import drive
drive.mount('/content/drive')
```

这将显示一个 URL，您需要点击该 URL 以授权访问。

为了确保我们的所有输入和输出都使用 UTF-8 进行解码和编码，我们设置了以下环境变量:

```
!export PYTHONIOENCODING=UTF-8
```

当这些完成后，是时候将预先训练好的 GPT-2(或者更确切地说，是它的脑叶切除版本)加载到内存中了。从`nshepperd`精彩的存储库中运行以下脚本:

```
!python3 download_model.py 345M
```

参数`345M`告诉脚本下载 3.45 亿参数版本的 GPT-2。接下来，是时候加载将用于微调的训练数据了:

```
!cp -r /content/drive/My\ Drive/data/rap_training_data.txt /content/gpt-2/rap_training_data.txt
```

确保根据您上传培训数据的位置调整 Google Drive 路径。

现在，我们需要做的就是运行来自`nshepperd`的培训脚本:

```
!PYTHONPATH=src ./train.py --dataset /content/gpt-2/rap_training_data.txt --model_name '345M'
```

如果你能阅读 Python 代码，我鼓励你去库[https://github.com/nshepperd/gpt-2/blob/finetuning/train.py](https://github.com/nshepperd/gpt-2/blob/finetuning/train.py)看看，研究代码，看看到底发生了什么。

基本上，数据集被转化为输入/输出对，其中输入是一系列标记，输出是序列中下一个标记的概率。传统的 **RNN** 方法包括对单个字符的训练(训练非常慢，但可以处理任何大小写和格式的变体)和对整个单词的训练(快得多，但更难适应大小写和格式，完全无法想出新的酷 hiphop 表达式，*yo furr reaaalllzz*)。相反，GPT-2 训练数据的预处理试图将文本数据压缩成包括任何语法或文体意义的最短可能字节组合。这是一个非常好的妥协！

在训练期间，从我们的数据集中生成样本，并馈送给 [Tensorflow](https://www.tensorflow.org/) ，Tensorflow 使用采样的样本来更新预训练的 GPT-2 模型的权重。

每隔 100 个时期，模型将保存一个检查点并生成一些随机输出。随着时间的推移，生成的样本应该看起来越来越符合训练数据，而不是任何内容的直接副本(如果发生这种情况，则模型[过度拟合](https://en.wikipedia.org/wiki/Overfitting))。

当你认为你已经完成了训练，只需在 Google Colab 中中断内核。如果您想稍后继续培训，请确保将您的检查点保存到 Google Drive:

```
!cp -r /content/gpt-2/checkpoint/ /content/drive/My\ Drive/rapping_ai
```

`nshepperd`方便地包含了一个脚本，通过给**定制输入并观察输出，从模型中交互地采样**。只需将微调后的模型复制到适当的文件夹中，然后运行脚本:

```
!cp -r /content/gpt-2/checkpoint/run1/* /content/gpt-2/models/345M/!python3 src/interactive_conditional_samples.py --top_k 40 --model_name "345M"
```

在训练了~2000 个纪元后，我将来自瑞典的著名说唱歌手**容·莱恩**喂给了这样的模型:`<<Yung Lean`(既然这是我们训练数据的格式，参见 [Part 1](/@max.y.leander/create-a-rapping-ai-using-deep-learning-part-1-collecting-the-data-634bbfa51ff5) )，并期望模型以容·莱恩的风格填写一首歌曲的标题和一些歌词。这是:

> <<yung lean="" i="" will="" rock="" you="" like="" a="" nip="">></yung>
> 
> [D-Skeet & D . c .(吹牛老爹&里克·罗斯)制片]
> 
> [第一节:容瘦]
> 
> 你最好相信我
> 
> 她不仅仅是一个你可以交往的女孩
> 
> 你可以和我的女孩在一起，女孩
> 
> 她就像是我的爱人
> 
> 她让我的整个世界感到渺小
> 
> 她让我的生活如此快乐
> 
> 作为一个女孩，我很幸运
> 
> 因为我会和你一起滚动
> 
> [合唱:容精益]
> 
> 我有很多情人
> 
> 我对我爱的人有很多爱
> 
> 我有他们无法忍受或憎恨的人
> 
> 他们觉得我们是一家人
> 
> 这是一种如此深沉的爱
> 
> 我很自豪我要给你一个(吉他即兴演奏)
> 
> [第二节:容瘦]
> 
> 我来自一个非常富有的国家
> 
> 我想给你我需要的东西，你知道(磁带卷)
> 
> 我来自一个非常富有的国家
> 
> 我和我所有的兄弟们去同一所学校
> 
> 我要去读研，我想得到全世界
> 
> 我不知道你想给我什么
> 
> 但是我必须放弃
> 
> 因为这是太多的爱，太多的爱
> 
> [合唱:容精益]
> 
> 我有很多情人
> 
> 我对我爱的人有很多爱
> 
> 我有他们无法忍受或憎恨的人
> 
> 他们觉得我们是一家人
> 
> 这是一种如此深沉的爱
> 
> 我为我所做的事情感到骄傲
> 
> 我把它放在架子上了
> 
> 因为这是一种如此深沉的爱
> 
> 我很自豪我要给你一个(吉他即兴演奏)
> 
> [第三节:容瘦]
> 
> 对我爱的人有很多爱
> 
> 对我爱的人有很多爱
> 
> 这都是爱
> 
> 这都是爱
> 
> 这都是爱
> 
> 这是，一切(吉他即兴重复)

相当不错！

事实上，我不知道这是否会通过作为一个真正的容精益歌曲。但至少它把一首说唱歌曲的基本结构搞对了。而爱情这个话题，绝对是贯穿整首歌的。

现在继续**创建你自己的数据集**(就像我在[第 1 部分](/@max.y.leander/create-a-rapping-ai-using-deep-learning-part-1-collecting-the-data-634bbfa51ff5)中所做的那样)并在你感兴趣的领域中生成真正新的东西**！**

# 我们今天学了什么？

*   如何加载一个名为 GPT-2 的预训练通用语言模型
*   如何使用我们自己的数据集针对特定目的微调该模型

# 下次我们会学到什么？

*   如何从文本合成语音
*   如何将音节与固定节拍匹配
*   使用深度学习将音频风格转移到您最喜欢的说唱歌手

我意识到下一课的目标比这个系列中迄今为止所做的要更难实现，但是缓慢而稳定地赢得比赛！

换句话说，敬请关注…