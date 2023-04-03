# 自然语言处理的发展—第 4 部分—变压器— BERT、XLNet、RoBERTa

> 原文：<https://medium.com/analytics-vidhya/evolution-of-nlp-part-4-transformers-bert-xlnet-roberta-bd13b2371125?source=collection_archive---------2----------------------->

![](img/86e92f6f184995fcc5b77d81a4826505.png)

## 使用 SOTA 变压器模型进行情感分类

这是终局！Transformers 是当今使用的主要深度学习架构之一，结合迁移学习来处理各种 NLP 任务。我们继续寻找情感分类任务的最佳解决方案。我们已经看到了基本技术— [TF-IDF 和单词包](/analytics-vidhya/evolution-of-nlp-part-1-bag-of-words-tf-idf-9518cb59d2d1)，然后我们看到了 [LSTMs](/analytics-vidhya/evolution-of-nlp-part-2-recurrent-neural-networks-af483f708c3d) ，接下来我们应用了[迁移学习和 LSTMs](/analytics-vidhya/evolution-of-nlp-part-3-transfer-learning-using-ulmfit-267d0a73421e) ，现在我们在这里探索一个完全不同的架构！

在我们继续之前，了解一下 rnn/lstm 的问题是很重要的。

# LSTMs 的问题

LSTMs 在 NLP 的发展过程中非常关键。这些是解决 rnn 内部问题并使深度学习更加广泛传播的一些最关键的架构。但是他们提出了一些问题，使他们很难工作。

1.  **慢练！**rnn 训练起来已经比较慢了，因为我们需要在训练时顺序地**提供数据**，而不是并行地**。隐藏状态将需要来自所有先前单词的输入才能取得任何进展。这种架构没有利用当今针对并行处理而优化的 GPU。再加上 LSTM 以多门的形式增加了复杂性，训练起来就更慢了。**
2.  ****更好的情境意识！**正常的 LSTMs 只在一个方向上处理单词，这限制了网络的上下文感知。即使是双向 LSTMs 也可以在向前&向后的方向上学习上下文并将它们连接起来，而更好的方法是同时查看向前和向后的单词！**
3.  ****长序列！**我们已经知道，lstm 由于其门的原因可以比 rnn 执行得更好，但即使如此，当我们尝试处理大量句子的任务时，lstm 的改进并不显著，例如，文本摘要和问题&答案。**

**有了**变形金刚**，让我们看看如何解决这些问题。**

# **变形金刚(电影名)**

**这种架构是第一次展示——你所需要的只是关注！(2017)论文。完整的架构可以分为两部分——编码器和解码器。在附图中，左边的结构是编码器，右边的是解码器**

**![](img/e00c9ef72b4ca0d10453f75b3f0f9a3a.png)**

**来自注意力的编码器-解码器架构是你所需要的！(2017)论文**

## **1.编码器**

**简而言之，编码器试图理解输入句子的上下文，并在此过程中学习语言是什么！让我们看看数据是如何在编码器中移动的。**

*   **编码器的输入是一个句子——比如大红狗。这些单个单词首先被标记化，并被在**输入嵌入**层中预先训练的单词嵌入所替换。**
*   **下一步，我们给单个单词向量添加一个**位置编码**。本质上是映射到单词在句子中的位置的函数。我们将它添加到我们的初始输入嵌入中，以确保除了单词之外，我们还嵌入了它的位置方面，同时我们将句子传递给编码器。**
*   **接下来，我们将它传递给**自我关注层**。这里的想法是创建一个每个单词的重要性矩阵，同时为同一个句子中的其他单词定义上下文。很明显，每个单词对它自己，然后对其他单词都有重要意义。**

**![](img/11f2872516522bd10ab03173d3783012.png)**

**红色越深，每个单词与所有其他单词的关注度越高。—图片来自变形金刚解说—[https://www.youtube.com/watch?v=TQQlZhbC5ps](https://www.youtube.com/watch?v=TQQlZhbC5ps)**

*   **最后，我们让它通过一个**前馈**层，这基本上是一个密集的神经网络层。**

**这个网络最大的好处就是我们不需要顺序传递数据而是并行传递！这大大提高了训练时间。而且由于，我们是以一种整体的方式来看待单词，没有任何方向感，对语言或上下文的整体理解要比其他架构好得多。**

**要理解 BERT 和 XLNet，只需要了解编码器就可以了。所以，如果这是你的目标，请跳到关于伯特的部分。接下来我将尝试解释解码器。**

## **2.解码器**

**简而言之，一个解码器将输出的句子作为输入，并试图输出下一个单词。将整个 Transformer 架构想象成一个语言翻译模型——输入为英文的“The big red dog ”,输出为法文的“le 格罗 Chien rogue”。现在，我们来看看这种输出在解码器网络中是如何流动的。**

*   **第一步类似于编码器。嵌入，结合每个输出字的位置编码。**
*   **接下来，我们计算**被掩盖的自我关注**。这与之前略有不同，因为这里我们有意地掩盖了即将到来的词语。本质上，在计算重要性时，我们不提供接下来应该出现的单词。这很有趣，因为这与我们在编码器中所做的不同。直观地说，在预测下一个法语单词时，我们可以使用所有英语单词的上下文，但我们不能使用**下一个**法语单词，因为这样一来，模型只会输出那个法语单词。我知道这有点令人困惑，但是当我们看到这些模型在实践中是如何使用的时候，它就会变得清晰。**
*   **然后，来自解码器网络的输入与编码器的输入一起被查看。在这里，我们试图一起建立英语和法语单词之间的上下文或关系，但是再次对法语单词进行掩蔽**
*   **最后，当单词通过前馈和线性层时，我们到达 Softmax。softmax 基本上是针对词汇表中的所有单词，它为我们提供了字典中所有单词的下一个单词的概率得分。为每个位置选择概率最高的单词。**

**请注意，这个过程也是并行发生的，类似于编码器。**

**我希望这能让你对变压器的体系结构有所了解，以及它们如何能比 LSTMs 有所改进。让我们试着看看两个这样的架构&它们的详细实现。**

# **伯特**

**Bert 代表来自变压器的**双向编码器表示。顾名思义，这种架构使用变压器网络的编码器部分，但不同之处在于多个编码器网络一个接一个地堆叠在一起。让 BERT 脱颖而出的另一个重要方面是它的训练方法。让我们试着理解这一点。****

## **1.预培训**

**这是模型理解什么是语言和上下文的阶段。对于这一部分，BERT 使用两个同时进行的任务来训练—**

*   ****掩蔽语言建模** —直观上，这就像一个“填空”的学习任务。该模型随机屏蔽句子的一部分，其工作是预测那些被屏蔽的单词。例如—输入为“懒狗之上的 ***【面具】*** 褐狐 ***【面具】*** ”，输出将是[“quick”、“jumped”]。**
*   ****下一句预测** —这里 BERT 取两句话，判断第二句话是否跟在第一句话后面，基本上就像一个分类任务。**

**简而言之，这两项任务让伯特理解了一个句子和多个句子中的语言和语境。**

## **2.微调**

**现在，模型基本上理解了语言和上下文是什么，接下来是为我们的特定任务训练它。在我们的例子中，这就是情感分类。根据我们的具体任务，我们只需在网络末端添加密集层即可获得输出。**

**让我们从实现 BERT 开始。我们将看看熟悉的 fast.ai 构造，但是由于这些模型在 fast.ai 中不直接可用，我们将不得不为 fast.ai 创建标记化器、词汇和数值化器的类。这非常令人困惑，尤其是如果不熟悉这些库的话。**

> **注意，下面的实现是受同一作者的[这篇](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta)优秀内核教程和 Medium 博客的启发。这不仅允许您快速实现转换器，而且在尝试其他强大的架构时也提供了极大的灵活性。**

**也就是说，我会试着在下面分享我对这个教程的解读。请随意查看[这个内核](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta)以获得更好的清晰度。关于这一点还有其他教程，我将在下面链接，但如果你想找到一个通用的解决方案来加载多种类型的变形金刚模型，我推荐阅读[这个实现](https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta)。**

**除了我们在上一个教程中学习的 **fast.ai** 库，这里我们将额外使用 **HuggingFace Transformers** 库。这个库拥有几乎所有主要的 SOTA 自然语言处理模型，以及用于自然语言处理任务的特定模型，如问答、文本摘要、掩码 LM 等。我们将在这里使用**序列分类**模型，但是也可以尝试其他任务的模型。**

**现在，使用 HuggingFace 库运行任何模型都需要你加载 3 个组件**

1.  ****模型类** —这将帮助我们加载特定模型的架构和预训练权重。**
2.  ****令牌化器类** —这将有助于将数据预处理成令牌。此外，在不同的模型中，句子的填充、开头和结尾以及对词汇中缺失单词的处理都有所不同。**
3.  ****配置类—** 这是存储所选模型配置的配置类。它用于根据指定的参数实例化模型，定义模型架构。**

**注意，我们需要为**相同的模型加载这三个类。**例如，在我们最初的试验中，如果我运行的是 BERT，我们为模型类加载**BertForSequenceClassification**，为标记化器类加载**bertokenizer**，为配置类加载 **BertConfig** 。**

**下一步是标记化！**

## **标记化**

**注意，BERT 有自己的词汇表和标记器。因此，我们需要围绕 BERT 的内部实现创建一个包装器，以便它与 fast.ai 的实现兼容。这可以通过下面所示的 3 个步骤来完成。**

1.  **首先，我们创建一个 tokenizer 对象，为我们的特定模型加载默认的 tokenizer。**

```
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
```

**![](img/151a0c9d64f6429595bce07766958616.png)**

**[标记器输出—图片来自作者](https://www.kaggle.com/jainkanishk95/evolution-of-nlp-4-bert)**

**2.然后我们创建自己的 **BertBaseTokenizer** 类，在这里我们更新了 ***tokenizer*** 函数，合并了帮助处理我们特定的一组变形金刚模型的函数。**

```
class BertBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_typedef __call__(self, *args, **kwargs): 
        return selfdef tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
        tokens = [CLS] + tokens + [SEP]
        return tokens
```

**我们用这个初始化我们的基础记号赋予器—**

```
bert_base_tokenizer = BertBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
```

**3.我们还没有完成，这是最令人困惑的部分。我们将我们的 **bert_base_tokenizer** 传递给 **Tokenizer** 函数，然后由 fast.ai 处理。这个额外的步骤很重要，所以请确保在您的实现中也这样做。**

```
fastai_tokenizer = Tokenizer(tok_func = bert_base_tokenizer)
```

**4.要在加载数据时使用它，建议将其转换为 TokenizerProcessor。我们可以在 DataBunch 调用中调用它，就像我们在前面的教程中看到的那样**

```
tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)
```

## **数字化**

**这一步包括将令牌编码成数字编码。同样，每个 transformer 模型都有自己版本的词汇表，因此也有自己的编码。让我们将它们也加载到熟悉的 fast.ai 框架中。**

1.  **首先，让我们在 fast.ai 中创建我们的 **Vocab** 库版本，更新它的函数— **numericalize** (将标记转换为编码)和 **textify** (将编码转换为标记)。在这些函数的定义中，我们分别使用—**convert _ tokens _ to _ ids**和 **convert_ids_to_tokens** 函数，它们与 HuggingFace 的预训练变压器模型一起工作。**

**![](img/f6d5d32ffe0db255a849e9ae172098b6.png)**

**[TransformersVocab 类定义—图片来自作者](https://www.kaggle.com/jainkanishk95/evolution-of-nlp-4-bert)**

**2.最后，我们将它传递到一个数值化处理器类中，类似于标记化处理器，我们将在创建 DataBunch 时调用它。**

```
transformer_vocab = TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)
```

## **加载数据**

**接下来，我们使用 DataBlock API 加载数据。**

```
databunch = (TextList.from_df(train, cols=’user_review’, processor=transformer_processor)
 .split_by_rand_pct(0.1,seed=seed)
 .label_from_df(cols= ‘user_suggestion’)
 .add_test(test)
 .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))
```

## **建模**

**在我们开始建模过程之前，我们使用预先训练好的模型作为基础创建自己的模型，只提取我们的特定预测所需的 logits。**

**![](img/462194f0633105c675a9b3273f2b9023.png)**

**[自定义分类模型—来自作者的图像](https://www.kaggle.com/jainkanishk95/evolution-of-nlp-4-bert)**

**最后，初始化附带了 HuggingFace Transformers 库的 **AdamW** 优化器。**

**![](img/9398a65a64660664bea31ce09e9367f7.png)**

**[学习者——来自作者的形象](https://www.kaggle.com/jainkanishk95/evolution-of-nlp-4-bert)**

**之后，我们可以直接运行模型，但在使用 fast.ai 库时引入另一个有用的工具可能会有所帮助。**

## **逐步解冻**

**在之前的教程中，我们介绍了判别微调和预定三角学习率。逐步解冻也是你可以探索，而建立模型。**

**这个想法很简单——像 BERT 这样的模型的早期层大多是为了理解语言——就像预先训练好的 CNN 模型一样，它们可以保持不变，只需调整最后几层就可以很快得到好的结果。**

**对于 fast.ai 中可用的模型(如 AWD-LSTM)，这些层已经以组的形式存在，这可以被视为不可训练的。对于 BERT，我们将不得不自己创建组。**

```
list_layers = [learner.model.transformer.bert.embeddings,
              learner.model.transformer.bert.encoder.layer[0],
              learner.model.transformer.bert.encoder.layer[1],
              learner.model.transformer.bert.encoder.layer[2],
              learner.model.transformer.bert.encoder.layer[3],
              learner.model.transformer.bert.encoder.layer[4],
              learner.model.transformer.bert.encoder.layer[5],
              learner.model.transformer.bert.encoder.layer[6],
              learner.model.transformer.bert.encoder.layer[7],
              learner.model.transformer.bert.encoder.layer[8],
              learner.model.transformer.bert.encoder.layer[9],
              learner.model.transformer.bert.encoder.layer[10],
              learner.model.transformer.bert.encoder.layer[11],
              learner.model.transformer.bert.pooler]
```

**这些层可以分成单独的组——**

```
learner.split(list_layers)
```

**现在，冻结所有的层，除了最后一层，我们使用——**

```
learner.freeze_to(-1)
```

**并且，让模型训练 1 个纪元。**

**之后，我们依次解冻另一层——**

```
learner.freeze_to(-2)
```

**而且，让模型训练为另一个时代。**

**最后，我们解冻所有层，让它运行 5 个纪元。**

## **准确度— 92%**

**这让您对使用 BERT 进行情绪分析有了初步的了解。但是，HuggingFace Library 并不仅限于 BERT。已经出现了几个新的模型，这些模型显示了对 BERT 的改进。**

# **XLNet**

**XLNet 在大约 20 个 NLP 任务中击败了 BERT，一举成名，有时利润率相当高。那么，什么是 XLNet，它与 BERT 有何不同？XLNet 的体系结构类似于 BERT。不过，最大的区别在于它的培训前方法。**

*   **BERT 是一个基于自动编码(AE)的模型，而 XLNet 是一个自回归(AR)模型。这种差异体现在 MLM 任务中，在该任务中，随机掩蔽的语言令牌将由模型预测。为了更好地理解这种区别，让我们考虑一个具体的例子[纽约，纽约，是，一个，城市]。**
*   **假设 BERT 和 XLNet 都选择两个代币[New，York]作为预测目标并最大化*日志(New，York | is，a，city)* 。此外，假设 XLNet 对因式分解顺序[是，a，city，New York]进行采样。在这种情况下，BERT 和 XLNet 分别约简为以下目标函数:**

> *****J{BERT} = log(New | is，a，city) + log(York | is，a，city)*** and**
> 
> *****= log(New | is，a，city) + log(York | New，is，a，city)*****

*   ***请注意，XLNet 能够捕获对(New，York)之间的依赖关系，而 BERT 忽略了这一点。虽然在这个例子中，BERT 学习了一些依赖对，如(New，city)和(York，city)，但很明显，XLNet 总是在给定相同目标的情况下学习更多的依赖对，并包含“更密集”的有效训练信号。***

***让我们看看 XLNet 是否提供了任何显著的改进。***

***总的来说，实现是相当相似的，在句子的填充、开始和结尾有微小的变化。您可以在这个内核中研究代码***

## ***准确度— 93%***

# ***罗伯塔***

***稳健优化的 BERT 方法(RoBERTa)是对 BERT 的再训练，具有改进的训练方法、1000%以上的数据和计算能力。***

***为了改进训练过程，RoBERTa 从 BERT 的预训练中移除了下一句预测(NSP)任务，并引入了动态屏蔽，使得屏蔽的令牌在训练时期期间改变。还发现较大的批量训练规模在训练过程中更有用。***

***重要的是，RoBERTa 使用 160 GB 的文本进行预训练，包括 16GB 的书籍语料库和 BERT 中使用的英语维基百科。***

***总的来说，实现是相当相似的，在句子的填充、开始和结尾有微小的变化。您可以在这个内核中研究代码。***

## ***准确度— 94%***

***通过几行代码，我们能够实现和研究 SOTA 变压器模型。我希望这一系列文章是您学习这些技术的良好开端。***

***有了这些新的创新，在过去的几年里，NLP 中可以做的事情的可能性发生了巨大的变化。这些变压器模型有可能取代所有现有的 LSTM/RNN 模型！他们没有放慢脚步——就在最近——open ai 的 GPT-3 发布了，它也是基于变压器架构，但使用解码器构建。与这些新模型/架构保持同步非常重要，并在此过程中不断学习！***

***下一篇博客再见！快乐学习:)***