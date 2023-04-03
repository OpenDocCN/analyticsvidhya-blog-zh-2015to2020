# 张量流中的自然语言处理

> 原文：<https://medium.com/analytics-vidhya/natural-language-processing-in-tensorflow-4bc8e1fba3f4?source=collection_archive---------19----------------------->

![](img/a992782760db456f3f88fbb525cb91ac.png)

[https://www.google.com/url?sa=i&source = images&CD =&CAD = rja&uact = 8&ved = 2 ahukewjsold-8 ellahuih 7 cahue 2 ahwqjhx 6 bagbeai&URL = https % 3A % 2F % 2fgithub . com % 2ft ensorflow % 2ft ensorflow&psig = aovvaw 1 aidt Q0 S6 wmeen L2 mj 0 NID&ust = 1573586855](https://www.google.com/url?sa=i&source=images&cd=&cad=rja&uact=8&ved=2ahUKEwjsoLD-8eLlAhUIH7cAHUe2AHwQjhx6BAgBEAI&url=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftensorflow&psig=AOvVaw1AIDtq0S6wmeeNl2MJ0Nid&ust=1573586853331843)

大家好，在这篇博客中，我将使用 TensorFlow 来讨论 NLP 概念。我在 Coursera 的实践课程 TensorFlow 的帮助下写这篇博客。这是初学者了解 TensorFlow 的最佳课程之一。

![](img/3cbaeadc26c31012dd2ef02537b0a8f1.png)

在这里，我们将了解如何使用神经网络建立基于文本模型的分类器。首先，我们将从查看文本中的情感开始，并学习如何建立模型来理解经过标记文本训练的文本，然后可以根据他们看到的内容对新文本进行分类。

对文本集中的每个字符进行字符编码，如 ASCII，这是一种常见的字符编码方法，但只用字母训练神经网络可能是一项困难的任务。当我们将数据输入神经网络时，为句子中的每个单词赋值是有意义的

![](img/bcadaa09a448bfc437865ced67de2d09.png)

单词编码

所以这至少是一个开始，以及我们如何开始训练一个基于单词的神经网络。TensorFlow 和 Keras 给了我们一些 API，让我们可以非常简单地做到这一点。

将图像输入神经网络很容易，因为像素值已经是数字了。让我们看看文本会发生什么？我们如何通过句子和文本实现这一点？

现在让我们看看这个概念:

我把这篇文章分成四个部分，分别是文本中的情感，单词嵌入，序列模型和序列模型&文学。

# 文本中的情感

有很多方法可以使用 tensorflow 和 keras 对单词进行编码。但是我们在这里要讨论的是**记号赋予器。**

![](img/5f11dfa7e68efd533a61ef2b79caf3c2.png)

请注意，在第一句话中，我将“I”大写，因为它是句子的开头；在第三句话中，在单词 dog 之后，我添加了一个感叹号。因此，通过设置 num_words 超参数，分词器要做的是按体积取前 100 个单词，并对它们进行编码。然后，标记器的 fit on texts 方法接收数据并对其进行编码。Tokenizer 提供单词索引属性，该属性给出包含单词和该单词的键值的字典。

您可以从输出中看到，在第一个句子的前面，我是大写的，但在这里是小写的。请注意，我们在第三句话中没有获得感叹号的键值。这是记号赋予器为你做的另一件事。它去掉了标点符号。

下图显示了有助于根据标记将句子转换为值列表的代码..

![](img/c3a1ad0f7c6e3643c7fb562cecd06828.png)![](img/eb02f6fe18234d030266a9898d265d02.png)

我们制作了一个字典，其中包含键值和属于该令牌的单词。现在下一步将是把你的句子变成基于这些记号的值的列表。在列出清单后，我们需要将每个句子处理成相同的形状，否则很难用它们来训练神经网络。当我们处理图像时，我们给定定义为输入层的图像的输入大小，这就是我们输入到神经网络的图像的大小。如果大小不同，我们将重塑图像，以适应同样我们要做的文本。

在上面的测试数据中，句子“我的狗爱我的海牛”被编码成[1，3，1]，这段代码的输出将是“我的狗我”,这是完全错误的，因为每当输入一个看不见的单词时，我们就添加一个属性，用“<oov>”token izer 构造函数填充它。</oov>

第一句话会是，我爱我的狗。第二个将是，我的狗 oov，我的 oov 在语法上仍然不是很好，但它做得更好了。随着语料库的增长，索引中的单词越来越多，希望以前看不到的句子能有更好的覆盖率。

**填充**

接下来是填料。正如我们之前提到的，当我们建立神经网络来处理图片的时候。当我们将它们输入网络进行训练时，我们需要它们大小一致。通常，我们使用生成器来调整图像的大小。在使用文本训练之前，你会遇到类似的要求，我们需要有一定程度的统一大小，所以填充是你的朋友。

下面是处理填充的完整代码。首先，您需要导入填充序列。这里你可以通过 maxlen 方法定义你需要多长的长度。

![](img/25356fa040adca1d6018e82ada2dc01d.png)

首先，为了使用填充函数，你必须从**tensor flow . keras . preprocessing . sequence .**导入`pad_sequences`

输出非常直接，您可以看到句子列表已经被填充到度量中。每一行都有相同的长度，这是通过在句子前面加零得到的。

在即将到来的博客中，我将讨论单词嵌入。