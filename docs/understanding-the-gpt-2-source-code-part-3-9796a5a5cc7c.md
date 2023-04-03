# 了解新 GPT 协议源代码第 3 部分

> 原文：<https://medium.com/analytics-vidhya/understanding-the-gpt-2-source-code-part-3-9796a5a5cc7c?source=collection_archive---------0----------------------->

![](img/e65a421434f0bd70afc8c5a9e1754d1c.png)

嗨！这是研究 GPT-2 的源代码的继续。你可以在这里找到第一部分和第二部分[，在这里](/@isamu.website/understanding-the-gpt-2-source-code-part-1-4481328ee10b)找到[。](/@isamu.website/understanding-the-gpt-2-source-code-part-2-4a980c36c68b)

在这里，我将在研究 sample.py 和 model.py 的同时，尝试介绍 GPT-2 的模型是如何工作的

# 样本序列

sample.py 的主要功能是在给定条件/输入的情况下生成文本输出。这是由 sample.py 中的 sample_sequence 完成的。

# 输入

样本序列的输入如下所示

```
def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0):
```

*部分只是强制函数的用户直接指定参数。例如，给定函数

```
def a(c):
    print(c)
def b(*,c):
    print(c)
```

而第一个函数可以被称为

```
a(“hi”)
```

并且只输出 hi，第二个函数必须被调用为

```
b(c=”hi”)
```

才能得到同样的结果！

```
if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)
```

正如我们在第 1 部分中看到的，这部分是针对 generate _ unconditional _ samples . py 和 interactive _ conditional _ samples . py 的，如果我们要生成无条件样本(没有输入的样本)，输入文本，上下文将被设置为一个由 start 标记初始化的张量。然而，否则，传入的编码文本将作为输入给出！在这里，在重新检查 interactive _ conditional _ samples . py 的代码后，我发现 OpenAI 没有决定在作为输入给出的传入文本的开头添加一个开始标记，这很有趣。相反，他们只是将其编码为

```
raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
```

我发现这很有趣，因为我认为你总是需要一个启动令牌！

# 为什么要有启动令牌？(知道的可以跳过！)

我认为在阅读第 1 部分为什么我们必须有开始标记时可能会有点困惑。所以，我在这里试着解释一下！开始标记，顾名思义，表示文本的开始。例如，对于文本“我很高兴”，开始标记在前面，结束标记在后面，并以“<start_token>我很高兴<end_token>”结束。我们这样做的原因是，当我们加载一个文本时，每个文本都有不同的长度！</end_token></start_token>

所以，在我们把字符串编码成数字之后，这里通常的方法是在末尾加 0。例如，如果 I 是 1，am 是 2，happy 是 3，那么它将被编码为 1230000…然而，这种方法的一个问题是，机器本身在学习序列的过程中，会对文本的开始和结束位置感到困惑。

例如，如果大多数文本都很短，并且有一个长的传入文本，如 1111111111113452000..，那么很有可能由于机器仅在 11113452 给出的位置经历了 0，它将开始忽略这些数字，并且总体上导致训练的相当糟糕的结果！这就是开始标记被引入的原因。它们表示字符串的开始，因此机器学习算法知道文本开始的位置和结束标记，在文本的结尾表示文本结束的位置。

```
def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }
```

接下来，在 sample_sequence 函数中，定义了这个阶跃函数。

该函数的第一行调用模型文件中的模型函数，然后返回一个 tensors lm_outputs 字典，其中包含两个键“logits”和“present”。我还不知道它们是什么，但是当我们深入研究代码时，我确信我们会发现它们是什么。

# 什么是张量？(知道的可以跳过！)

对于不那么熟悉 Tensorflow 的人来说，我觉得“张量”这个词有点神秘。它就像是图形的一个组成部分。在 Tensorflow 中，当您编写以下代码时，

```
a = tf.constant(1)
b = tf.constant(1)
c = a + b
```

c 的值不会是 2。(TLDR；至少在 Tensorflow 2.0 更新之前没有，但由于这段代码是在此之前编写的，所以我忽略了它)

事实上，它的值直到运行时才会被正确设置。它唯一知道的是，c 是 a 和 b 相加后的值。

我们可以做的是设置 a 和 b 的值，以查看 c 的结果。为此，我们启动一个会话。张量流中的会话允许我们实际执行张量中的操作。所以，在这种情况下，我们可以通过做一个 session 来评估 c 的值。是这样做的。

```
with tf.Session() as sess:
    print(sess.run([c]))
```

和[2]应该打印出来。如果我们想给 a 和 b 输入新的值，我们可以这样做

```
with tf.Session() as sess:
    sess.run([c],{a:2,b:3})
```

和[5]将被输出。

# 查看 model.py 内部

现在我们知道了 model.py 的模型函数输出张量，让我们看看 model.py 是如何设置的，并可能看到一些正在使用的算法！

# 张量流观测仪

当我们看模型函数的顶部时，我们看到

```
def model(hparams, X, past=None, scope=’model’, reuse=False):
 with tf.variable_scope(scope, reuse=reuse):
```

Tensorflow 在这里定义的变量作用域主要是为了便于调试。当作用域中一个名为 x 的张量出错时，如果作用域名设置为 hello，那么错误会在错误消息中将该张量称为 hello/x。

此外，虽然我还没有在这个用例中使用它，但它也可以用于[共享变量](https://www.tensorflow.org/api_docs/python/tf/variable_scope)！这就是重用参数的用武之地。举一个文档中的例子，它可以以如下方式使用！

```
def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
assert v1 == v2
```

# 提取批量和序列长度

模型函数中的下一行是

```
batch, sequence = shape_list(X)
```

在这里，当我们回头看 sample.py 时，

```
def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
```

我们看到 X 被设置为称为令牌的步骤的输入。现在，让我们研究一下 shape_list 函数，看看它到底是做什么的。

```
def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
```

由于我不太清楚它是如何工作的，我进入了 idle(如果你安装了 python，只需在开始菜单中输入 Idle 就可以打开)并尝试了一些东西。

开始之前，我看了看这条线

```
return [dynamic[i] if s is None else s for i, s in enumerate(static)]
```

从中我们可以看出，如果静态的维度都不是 none，那么函数只是返回 x.shape.as_list()。因此，我做了一个张量如下

```
example = tf.placeholder("float", [None, 5])
```

这用一个非 5 次方的张量初始化示例。这里的“无”表示在会话开始之前，它可以接受任何维作为第一维。

我首先输出 example.shape.as_list()，它输出

```
[None, 5]
```

对于 tf.shape(示例)，

```
<tf.Tensor ‘Shape_5:0’ shape=(2,) dtype=int32>
```

被退回。返回的第一维

```
<tf.Tensor 'strided_slice_2:0' shape=() dtype=int32>
```

第二维度又回来了

```
<tf.Tensor ‘strided_slice_3:0’ shape=() dtype=int32>
```

当我把它传递给函数时，

[<tf.tensor shape="()" dtype="int32">，5]</tf.tensor>

被输出。起初，我不太明白这样做的用处，但我注意到的一件事是，这允许批量大小和序列长度，形状列表函数的输出可以在运行时设置，因为它是一个张量！

然而，我认为未来的 ML 学生需要注意的一件事是，我们仍然需要意识到，我们不能用未定义大小的权重和偏差来训练网络。

# 神经网络中最基本的概念(知道的可以跳过！)

神经网络中最基本的概念是矩阵乘法和矩阵加法。假设我们有一堆维数为(无，10)的数字作为输入。(TLDR，无一是批量大小)

假设我们想知道这十个数字是好是坏，用 0 还是 1 来表示。

虽然可能有许多方法可以做到这一点，但 ML 工程师应该想到的第一个模型是直接将 10 维缩减为 1 维。这是通过将输入乘以维度(10，1)的权重矩阵并添加维度(1)的偏差来完成的。(我可能会在另一篇文章中讨论线性代数，但我不认为我会在这里讨论)。输出是(无，1)

这允许维度按比例缩小，并且使用诸如梯度下降等技术来训练网络。然而，需要注意的一个重要方面是权重矩阵的维数和偏差是恒定的。这就是为什么他们是可训练的。因此，我们不能使用 OpenAI 的技术来秘密地将可变大小的维度设置为权重矩阵或偏差的维度之一。我决定写这个，因为，嗯，我犯了这些错误！

模型函数的下一行是

```
wpe = tf.get_variable(‘wpe’, [hparams.n_ctx, hparams.n_embd],
 initializer=tf.random_normal_initializer(stddev=0.01))
wte = tf.get_variable(‘wte’, [hparams.n_vocab, hparams.n_embd],
 initializer=tf.random_normal_initializer(stddev=0.02))
```

tf.get_variable 是可以训练的变量，初始值设定项是这些变量被设置的初始值。在这种情况下，他们将其设置为均值为 0，标准差为 0.01 或 0.02 的正态分布，我发现这很有趣，因为我倾向于将其设置为 0 或 1。现在我想起来，直觉上这是有道理的，但除了经过测试，它提高了性能之外，我不认为有什么合理的解释。反正我自己去试试。

虽然我不确定 wpe 和 wte 代表什么，但是我们可以研究 hparams 有什么值。

默认参数如下所示

```
def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )
```

n_embd 表示嵌入大小。

# 什么是嵌入(知道的可以跳过！)

嵌入基本上是一种将每个数字表示为向量的方法。这使得机器学习算法能够理解单词之间的异同。比如让我们看看猫狗 vs 车房子之类的词。由于猫和狗在单词方面非常相似，我们希望代表它们的向量比像 house 和 cat 这样的单词更接近。

嵌入大小给出了每个向量的大小，是 768，这是相当大的！

# 一个奇怪的属性

然而，默认参数中一个奇怪的属性是 n_vocab 是 0。当我们看 wte 是如何定义的，我们发现 n _ vocabs 在这个维度中

```
wte = tf.get_variable(‘wte’, [hparams.n_vocab, hparams.n_embd],
 initializer=tf.random_normal_initializer(stddev=0.02))
```

因此，这将永远是一个 0 维的张量吗？当我查看与模型一起保存的 hparams.json 文件时，这个问题很快得到解决，正如我们在下面看到的，参数是不同的

```
{
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}
```

# 设置输入

因此，我们现在可以合理地假设 n_vocab 是词汇量。虽然我不会进入 n_head 和 n_layer，但我认为假设 n_ctx 是上下文的最大长度是合理的，但是我们还不能确定！

```
wpe = tf.get_variable(‘wpe’, [hparams.n_ctx, hparams.n_embd],
 initializer=tf.random_normal_initializer(stddev=0.01))
wte = tf.get_variable(‘wte’, [hparams.n_vocab, hparams.n_embd],
 initializer=tf.random_normal_initializer(stddev=0.02))
```

因此，我们可以合理地确定 wte 是一个查找表，它保存了所有对应于令牌值的向量！

```
past_length = 0 if past is None else tf.shape(past)[-2]
h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))
```

现在，虽然这主要是一个猜测，我怀疑过去是模型到目前为止的输出。这主要是因为模型函数是从 samples.py 中的 step 函数调用的，所以我怀疑 step 函数是在每次模型输出新令牌并将该令牌作为输入添加到模型并再次调用时被调用的！这一点以后会得到证实。

虽然我不能说过去的形状，但从名称判断，past_length 应该包含到目前为止输出的文本的长度。

现在，让我们看看 h. tf.gather 是一个函数，它返回第一个参数的索引，这个索引是由第二个参数给出的。例如，如果 a 是一个张量

```
a = tf.constant([1,2,4])
```

要得到值 2，我们只需要调用

```
tf.gather(a, tf.constant([1]))
```

现在，正如我们在加法的第一部分看到的，

```
tf.gather(wte, X)
```

因为 x 是记号，wte 是将记号连接到向量的查找表，所以我们可以说这是到目前为止收集的记号的向量表示。让我们看看 wpe 部分，并试着弄清楚它是做什么的

```
tf.gather(wpe, positions_for(X, past_length))
```

为此，让我们看看 positions_for 函数。函数的位置给定为

```
def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)
```

首先要注意的一件有趣的事情是，与模型函数不同，batch_size 的值和步骤是这样直接获取的

```
batch_size = tf.shape(tokens)[0]
nsteps = tf.shape(tokens)[1]
```

我不知道他们为什么这样做，所以如果有人知道，请告诉我！

现在，在这之后，expand_tile 函数被这样调用，

```
expand_tile(past_length + tf.range(nsteps), batch_size)
```

扩展图块函数给定为

```
def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)
```

输入的参数是从 past_length 到 past _ length+序列长度或 x 的范围，而大小是批处理大小。

ndims 是张量的轴数。例如，如果它是一个二维张量，它是 2，如果它是一个三维张量，它是 3。

tf.tile 基本上将第一个参数扩展了第二个参数倍。你可以点击查看文档[。](https://www.tensorflow.org/api_docs/python/tf/tile)

这个平铺函数现在所做的是为所有批次堆叠一个范围从 past_length 到 past_length+sequence length 的批次！如果需要解释，请在评论里告诉我！

现在当我们回到

```
tf.gather(wpe, positions_for(X, past_length))
```

我们看到，从 past_length 到 past _ length+x 的序列长度的 wpe 的索引被采用。由于我首先不确定 wpe 是什么，我不能完全确定，所以我决定检查一下！我写了这个令人惊叹的博客。wpe 基本上做的就是告诉模型某个特定的单词在哪里！所以，如果它像第 5 个词，这个 wpe 会添加签名说这个词是第 5 个词，并添加它，这是相当耐人寻味的。这叫位置编码！

h 最终通过将表示记号的向量和位置编码相加而获得。

由于这篇文章比预期的要长一些，我想我会把下一篇文章的观点留到下一篇，因为我将进入变形金刚，这对于专家和初学者来说都是一个相当复杂的话题！

# 然后

如果你有兴趣，请查看下一篇文章[这里](/@isamu.website/understanding-the-gpt-2-source-code-part-4-a5fbb89e5038)！