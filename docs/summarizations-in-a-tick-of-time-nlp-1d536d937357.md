# 一瞬间的总结😲- NLP！

> 原文：<https://medium.com/analytics-vidhya/summarizations-in-a-tick-of-time-nlp-1d536d937357?source=collection_archive---------28----------------------->

![](img/779cacf677526987419953bcb8373b48.png)

本·怀特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

是啊！你没听错。

这里我们看到了如何在一瞬间获得一篇文章的摘要或给定的数据。使用自然语言处理，我们将充分利用各种模型，并使用 rouge 分数来查看这些模型的性能。

准备好了吗？？？😃

摘要技术主要分为两种类型，即提取摘要和抽象摘要。

*   在抽取中，我们识别文章的重要部分，并从原始数据中生成包含原始上下文含义的句子子集。
*   抽象地说，我们解释原始的上下文，并以一种新的可能的方式生成摘要，而不改变上下文的含义。

在自然语言处理领域，我们有各种模型来执行上述任务。让我们看看这些模型，也看看我们如何实现和赶上他们的胭脂分数。(评估总结任务的指标)

E **摘录:**

在 ES 中，我们使用 Bert、XLNet 和 GPT2 等模型来执行摘要任务。下面是如何编码...(以下是回购份额)

**伯特模型:**

```
**summarizer_bert = Summarizer()
summary_bert = summarizer_bert(data, min_length=30, max_length = 140)**
```

**XL 网型号:**

```
**summarizer_xlnet = TransformerSummarizer(transformer_type="XLNet",
transformer_model_key="xlnet-base-cased")**
```

**GPT 新协议模型:**

```
**TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")**
```

Ab**b 摘要:**

在 AS 中，我们使用 Bart 和 T5 等模型来执行总结任务。以下是如何在…上编码💻

**巴特模型:**

```
**summarizer_bart = pipeline(task='summarization', model="bart-large-cnn")
summary_bart = summarizer_bart(data, min_length=30, max_length = 140)**
```

**T5 型号:**

```
**summarizer_t5 = pipeline(task='summarization', model="t5-large")
summary_t5 = summarizer_t5(data, min_length=30, max_length = 140)**
```

从上面的代码中，我们知道了如何调用 transformer 模型来执行操作。但是在回购中，我们可以看到对上述任务的完整评估以及指标。随意克隆回购，物尽其用。(别忘了给 a⭐️if 你觉得有用的东西)

[](https://github.com/rahulmadanraju/Summarizers) [## rahulmadanaju/总结者

### 摘要是将文档的整个上下文缩短为一个或多个短句的过程。这里我们使用…

github.com](https://github.com/rahulmadanraju/Summarizers) 

上述总结工作的示例如下所示:👇

```
**data** = """
To do not love the blue eyes of kids is very difficult. If you are not born with the blue eyes, to obtain blue eyes naturally is not possible. But there are some tricks which you can use to make you feel that your eyes are blue. Or if you are really want to obtain blue eyes then you can obtain by surgery.Like the here and skin colour, our eyes colours is also genetic. This means that breaking without genetic code or cell structure you cannot change the eyes of your colour permanently.The eyes of your colour depend on the melanin which is present inside your eye. It depends on the amount of melanin. The less melanin your eyes have the more your eye will be blue and if melanin is more then you will get brown shade more.The amount of melanin is very less when a newly baby born. That is why you will find their eyes blue, in most of the newly born babies. Due to a genetic mutation, the colour of eyes varies a lot. You can find many type of colours in newly born babies.if you found that your eyes colour is changing automatically, specially from brown to blue, then you should immediately contact Dr specially eyes doctor. The change of colour in eyes could be due to many type of diseases. There are some chances that Some diseases can also make your blind either temporarily or permanently. The change of colour could be the sign of many diseases. The change of eye colour could be very exciting sometimes but if your eye colour changed naturally then there is no need to worry and consult a doctor. Because it is happened naturally so you don’t have to care about anything.However there are some places in the world where there is possible to change eye colour with surgery. They claim to change eye care . However results never proved. The experiment have not proved that changed eye colour remain to how many days and what does it effect on the health.Before going into the surgery you need to know what are the pros and cons of changing the eye colour through surgery. You have to take out some time and know merits and demerits very clearly before going into it. Also make sure to pass any tests before surgery.There are many type of laser surgeries in the market which can change the colour of your eye to blue. In the special type of surgery, Dr burn the lower layer of melanin in your eyes, which makes the iris appear to blue. Since this surgery test is new, we don’t have much information regarding affecting our health.
"""**Ouptut:** If you are not born with the blue eyes, to obtain blue eyes naturally is not possible. There are many type of laser surgeries in the market which can change the colour of your eye to blue. In the special type of surgery, Dr burn the lower layer of melanin in your eyes.
[{'rouge-1': {'f': 0.1920289837713322, 'p': 1.0, 'r': 0.1062124248496994}, 'rouge-2': {'f': 0.18545454374241324, 'p': 0.9807692307692307, 'r': 0.10240963855421686}, 
'rouge-l': {'f': 0.3083333307253473, 'p': 1.0, 'r': 0.18226600985221675}}]
```

如果你觉得这篇文章有用，请随意提问，也给我鼓掌👏如果你喜欢的话！

以下是我的其他博客:😑

[](/analytics-vidhya/nlp-pipelines-in-a-single-line-of-code-500b3266ac7b) [## 单行代码中的 NLP 管道

### 非常感谢拥抱脸变形金刚开源社区。因为它，我们正在最好地利用…

medium.com](/analytics-vidhya/nlp-pipelines-in-a-single-line-of-code-500b3266ac7b) [](/analytics-vidhya/is-accuracy-the-only-metric-of-evaluation-yes-and-no-6a65590ec39d) [## 准确性是评价的唯一标准吗？？？“是”和“不是”。

### 答案是肯定的，也是否定的。每个人都有自己的条件。

medium.com](/analytics-vidhya/is-accuracy-the-only-metric-of-evaluation-yes-and-no-6a65590ec39d) [](/@rahulmadan_18191/a-game-of-darts-in-bias-and-variance-3ed00a77b0f3) [## 一种有偏差和变化的飞镖游戏

### 你是机器学习爱好者，也想玩飞镖吗？嗯，正确的时间在正确的页面上。

medium.com](/@rahulmadan_18191/a-game-of-darts-in-bias-and-variance-3ed00a77b0f3) 

编码快乐！😎