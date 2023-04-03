# 疾病命名实体识别中的词嵌入比较

> 原文：<https://medium.com/analytics-vidhya/word-embedding-comparison-for-disease-named-entity-recognition-a99850653e1c?source=collection_archive---------2----------------------->

![](img/6703a419cf988eb64208ee5eef66b5d1.png)

[https://pix abay . com/vectors/scale-balance-weight-measure-2247163/](https://pixabay.com/vectors/scale-balance-weight-measure-2247163/)

这是一个命名实体识别(NER)任务的单词嵌入与疾病和不利条件的快速比较。这并不是详尽的，主要是因为想要尝试今年在健康自然语言处理任务中流行的 ELMo 嵌入。

# **命名实体识别:**

NER 在许多 NLP 任务中非常有用。大多数人能够立即理解的一点是能够区分苹果是什么类型的*苹果的股票今天下跌*和*苹果是一种很棒的零食*。NER 也非常善于判断几个单词是否可以组成几个单词，这些单词可能是名字，但实际上是一个名字。

有大量的文本医疗保健数据的宝库，NER 也可以在这里有用。它可以从快速编写的医生笔记中提取并正确分类高危疾病和症状的短语。首字母缩略词总是一个挑战，但我们可能能够决定笔记中写的 CA*高风险是指该州还是指癌症和癌*一种与皮肤有关的癌症。

# 数据:

我使用的疾病和副作用 NER 数据集:[https://www . scai . fraunhofer . de/en/business-research-areas/bio informatics/downloads/corpus-for-disease-names-and-Adverse-Effects . html](https://www.scai.fraunhofer.de/en/business-research-areas/bioinformatics/downloads/corpus-for-disease-names-and-adverse-effects.html)

# **嵌入:**

唯一没有在维基百科上训练过的单词嵌入是 EHR(电子健康记录)嵌入。单词的数量和单词向量的维数变化很大，这当然会影响性能。

那么什么是 ELMo 嵌入，为什么它们是特殊的？与大多数广泛使用的嵌入方法不同，ELMo(来自语言模型的嵌入)标记是根据它们所在的句子来上下文化的，而不仅仅是左右的标记。正如在 [ELMo 论文](https://arxiv.org/abs/1802.05365)中提到的，有类似的竞争算法，如 [context2vec](http://www.aclweb.org/anthology/K16-1006) 和 [coVe](https://arxiv.org/abs/1708.00107) ，ELMo 的作者显示这两种算法都优于可能进行直接比较的算法。论文中没有提到但值得一提的是 Facebooks InferSent 和谷歌的通用句子编码器。虽然它们关注的是句子，但后两者并不像 ELMo 那样关注单词级嵌入。基本上，一般的想法是获得隐藏在单词中的词汇层。

埃尔莫嵌入:[https://allennlp.org/elmo](https://allennlp.org/elmo)

手套嵌入(普通爬行(42B 令牌，1.9M vocab，未封装，300d 矢量):[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

# **结果:**

报告的所有分数都是疾病和不良反应实体的 F1-Micro on [BIO](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) 格式。实质上，如果两个实体被认为是一个整体，那么它们都会被评分。例如:如果*心力衰竭房颤【疾病】*是，但如果*心力衰竭房颤*不是。

f1-微是微平均精度和召回率的调和平均值(见下文)。这是一个合理的**(更多信息请见下文)**指标选择，因为它考虑了*不利*和*疾病*实体的类别不平衡，我们没有特定的指标需要优化:

> 微精度:(TP1+TP2)/(TP1+TP2+FP1+FP2)
> 
> 微回忆:(TP1+TP2)/(TP1+TP2+FN1+FN2)

埃尔莫嵌入(5.5b，200d):0.779±0.02

EHR/生物医学文本嵌入(约 3b 字，w2v cbow，200d):0.493±0.05

手套(42b，300d):0.811±0.04

手套(6b，50d):0.750±0.04

手套(6b，100d):0.780±0.01

手套(6b，200d):0.804±0.04

手套(6b，300d):0.816±0.03

快速文本(160 亿，300 天):0.791±0.05

结果(如果它们显示了什么的话)似乎表明这种比较是不公平的。也就是说，结果中最令人惊讶的可能是 EHR 嵌入的糟糕表现。这也许凸显了对更大语料库的需求。此外，这些都是使用 cbow word2vec 方法训练的，这不是我对罕见疾病单词的首选。这可能在与手套相比时得到反映，手套具有通过其共现频率来加权罕见词的能力。ELMo 嵌入的 AllenNLP 网站主持人确实指出，他们遗漏了与 GloVe 的比较，因为他们不认为它们是等价的比较。

尽管如此，F1 对于数据科学家来说是一个“安全的地方”,但对于你的 NER 任务来说，这是值得考虑的。事实证明，边界误差是生物应用中最大的误差来源之一。针对 F1 的优化可能会导致我们忽略*左翼*，因为当将*侧翼*标记为位置时，该短语被标记为*位置*，这仍然是部分且显著的成功。标签错误是最大的问题。

另一个需要考虑的问题是，对于你正在训练的语料库，实体的潜在稀疏性。大多数公开可用的生物医学文本数据集都是基于研究文章，甚至仅仅是摘要，这些摘要更密集地包含了将被识别为生物医学概念的对象。此外，这些数据集的基调更具学术性，可能无法捕捉某些疾病的重要指征。

dit:在我写这篇文章的时候，我偶然发现了脸书的元嵌入，这是一种机制，用来确定哪些嵌入在你的预测任务中最有效。[纸在这里。](https://research.fb.com/publications/dynamic-meta-embeddings-for-improved-sentence-representations/)也许这种集成型方法最有趣的用途之一是能够观察嵌入之间的特殊单词的变化。作者提出了一个论点，建模者不应该首先选择他们的嵌入，而是将它留给 DME(动态元嵌入)的客观严格性。

另一件要记住的事情是，许多这些通用、宽泛的标签并没有明确定义什么是数据清理、令牌化，以及它们具体针对什么进行优化。

# 延伸阅读:

如果你想阅读更多关于单词嵌入的最好和最新的内容，我发现[这篇](/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)很有帮助，也很有意义。

如果你想深入了解去年的发展趋势:[http://ruder.io/word-embeddings-2017/](http://ruder.io/word-embeddings-2017/)

本文中使用的 NER 数据集附带的论文:[生物医学文献中疾病和副作用识别资源的实证评估](http://www.nactem.ac.uk/biotxtm/papers/Gurulingappa.pdf)

感谢阅读！想法？评论？