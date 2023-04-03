# 一年回顾:2019 年的人工智能

> 原文：<https://medium.com/analytics-vidhya/a-year-in-review-ai-in-2019-bce4a3070ef9?source=collection_archive---------13----------------------->

![](img/5f905f7804f1e02192dc662206848de3.png)

2019 年见证了算法、数据、投资、研究论文等的激增。这是快节奏的一年，在新方法、服务和框架的研发方面取得了进展。以下是 2019 年人工智能各个领域的一些里程碑。

# 1.通用机器学习

为了简单起见，我将机器学习推广到这个等式，这或多或少适合许多情况，

> 机器学习=数据+算法+人在回路中

让我们从这个角度来看 2019 年的发展，

a)在数据端，

由于数据是新的石油，在许多情况下，我们会遇到数据短缺来训练我们的模型。为了解决这个问题，合成数据生成算法，如[生成对抗网络](https://arxiv.org/abs/1406.2661)和统计模型，使我们能够在缺乏训练机器学习模型所需数据的情况下生成所需数据(以数字、分类、文本、语音、图像或视频格式)。

b)在算法方面，

机器学习算法的实施和运作已经扩大，在这一点上，我们有令人惊讶的即插即用框架，如 numpy，pandas 和 scikit-learn。

人工神经网络是机器学习的一部分，令人惊讶的是，它是由生物神经网络启发的(所以我们基本上是由自己启发的！！)但幸运的是，创建人工神经网络的主要挑战之一是为它们选择正确的框架，截至目前，2019 年，数据科学家在各种选项中被宠坏了，如 [PyTorch](https://pytorch.org/) 、[微软认知工具包](https://docs.microsoft.com/en-us/cognitive-toolkit/)、 [Apache MXNet](https://mxnet.apache.org/) 、 [TensorFlow](https://www.tensorflow.org/) 等。但问题是，一旦神经网络在特定的框架上进行训练和评估，将它移植到不同的框架上是极其困难的。这在某种程度上削弱了机器学习的深远能力。因此，为了解决这个问题，AWS、脸书和微软合作创建了[开放式神经网络交换(ONNX)](https://onnx.ai/) ，它允许跨多个框架重用经过训练的神经网络模型。现在 ONNX 将成为一项基本技术，它将提高神经网络之间的互操作性。

c)人在回路中

为了促进人类在循环中的参与，数据标记服务和工具由亚马逊、Hironsan 等公司开发，以创建诸如[亚马逊机械土耳其](https://www.mturk.com/)(使您更容易构建标记数据的市场)和 [Doccano](https://doccano.herokuapp.com/) 等工具，这些工具有助于标记数据。

# 2.自然语言处理

自然语言处理(又名 NLP)是计算机科学的一个领域，人工智能专注于机器理解语言和解释信息的能力。

谈到 2019 年的 NLP，变形金刚模型一直在抢风头，因为 2019 年几乎每个月都有新模型出现，如伯特，XLNet，罗伯塔，厄尼，XLM-R 和阿尔伯特。

2017 年，谷歌发布了 [Transformer 模型](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)，打破了机器翻译的最先进成果。它通过引入自我关注作为流行的递归和卷积神经网络的替代方案，改变了深度学习的工作方式。2018 年， [BERT 模型](https://arxiv.org/abs/1810.04805)基于 Transformer 架构发布。该模型打破了许多不同 NLP 任务的最新记录，如文本相似性、问题回答和语义搜索。最近，中国百度的厄尼在理解语言方面比谷歌的前任伯特表现得更好。首先用于中文，现在它甚至更好地用于英语，这就是一年的结束方式！

当谈到文本生成时，同样值得注意的是最受关注的模型之一 [OpenAI 的 GPT-2](https://www.linkedin.com/pulse/text-generation-openais-gpt-2-production-emmanuel-raj/) 发布了。Open AI 发布了最大版本的 GPT-2(参数为 1.5B)。自己动手测试吧:[https://talktotransformer.com/](https://talktotransformer.com/)

# 3.计算机视觉

计算机视觉是人工智能系统像人类一样看到事物的能力，在过去几年中，它在所有领域都越来越受欢迎。计算机视觉技术的当前状态是由深度学习算法驱动的，该算法利用一种特殊的神经网络，即卷积神经网络(CNN)，来理解图像。今年，在医疗设备、食品和饮料、制药和汽车行业提供质量控制的计算机视觉解决方案激增。推动了最先进的[实时图像分割](https://arxiv.org/pdf/1903.08469v2.pdf)、[点云分割](https://arxiv.org/pdf/1908.08854.pdf)，图像生成和姿态估计。

此外，谷歌和 facebook 开发了[开源数据集](https://storage.googleapis.com/openimages/web/index.html) (Open images V5-约 900 万张图像的数据集，标注有图像级标签、对象边界框、对象分割遮罩和视觉关系)，以使用名为[皮媞亚](https://github.com/facebookresearch/pythia)的模块化框架将自然语言处理和计算机视觉结合起来，这在视觉问题回答任务中表现出色。这是皮提亚工作时的一瞥，

这里有一个皮媞亚启动并运行的[演示(由 CloudCV 提供),你可以试一试。](https://vqa.cloudcv.org/)

# 4.强化学习

强化学习是一种机器学习范式，其中学习算法不是基于预设数据而是基于反馈系统来训练的。这些算法被吹捧为机器学习的未来，因为它们消除了收集和清理数据的成本。强化学习已经在金融、网络安全、制造等行业取得了进展。

下面是关于[2019 年最佳深度加固研究的入门](/@ODSC/best-deep-reinforcement-learning-research-of-2019-so-far-e8e83a08c449)

# 5.自治系统

对自动化平凡而重复的机器学习任务的追求已经持续了一段时间，这导致了像 [AutoML](https://cloud.google.com/automl/) 、 [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/) 这样的工具，它们可以用于大规模训练高质量的定制机器学习模型。关于技巧和建议，请查看我的博客[构建健壮、可扩展和自动化的 ML 系统](https://www.linkedin.com/pulse/robust-scalable-ml-lifecycle-high-performing-ai-team-emmanuel-raj/)。

互联自动化车辆(CAV)是一项变革性技术，具有改变我们日常生活的巨大潜力。近年来，CAV 相关的研究取得了显著进展。特别是在 CAV 间通信、CAV 的安全性、CAV 的交叉路口控制、CAV 的无碰撞导航以及行人检测和保护方面。点击了解更多最新[技术。](https://www.sciencedirect.com/science/article/pii/S2095756418302289)

# 6.伦理人工智能

看到大公司强调人工智能的这一方面令人振奋。我想让你们注意一下这些公司发布的指导方针和原则:

*   [谷歌的人工智能原则](https://www.blog.google/technology/ai/ai-principles/)
*   [微软的人工智能原则](https://www.microsoft.com/en-us/ai/our-approach-to-ai)

这些本质上都是在谈论人工智能中的公平以及何时何地划清界限。此外，各种团体在这些领域取得了重大进展，如欧盟委员会和高级别专家组提出了可信人工智能的[道德准则](https://ec.europa.eu/digital-single-market/en/news/ethics-guidelines-trustworthy-ai)结果。作为欧洲人工智能联盟的一部分，我有幸在 2019 年 6 月 26 日在比利时布鲁塞尔举行的首届欧洲人工智能联盟上参与了这些准则的制定过程。

![](img/900464099dcdc1a2c594c04130b3a34f.png)

以下是我对欧洲值得信赖的人工智能的 8 个促成因素的想法。

正如你所看到的，2019 年发生了很多事情，跟上所有这些发展变得越来越困难，如果我错过了什么，请在下面的评论中告诉我，我很高兴听到并学习你的观点。

接下来的十年和 2020 年将会把界限推得更远，拥抱自己沉浸在大量的学习和改变中。希望，人工智能是好的。

干杯 2020，祝你新年快乐成功！