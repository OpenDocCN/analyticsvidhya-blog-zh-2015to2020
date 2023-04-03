# 自动答题评分器

> 原文：<https://medium.com/analytics-vidhya/automated-answer-grader-c65af94a145e?source=collection_archive---------18----------------------->

使用不同的向量聚类来学习答案和奖励分数。

![](img/cb1a9151fd35fc1b96e1a6d15c110af7.png)

教师手动评估学生的脚本

手动评估脚本是一个巨大的 Map-Reduce 系统。答案脚本首先在教师中分发。评价由每位老师完成。理事机构采取措施避免任何不想要的结果。最后，所有这些成绩都被收集起来，以供公布结果。想象一下，如果我们可以有一个系统，它可以自己接收答案脚本和评分，为每个答案生成一个反馈报告，所有这些都在一瞬间完成。我们的项目就是这样做的。

首先，我们需要准备一个能够理解用自然语言书写的答案的系统。要做到这一点，我们需要让一个系统学习。我们的数据集是一个 JSON 文件，格式如下。

```
{
    "Question": {
        "Answers: [
            "Answer 1",
            "Answer 2",
            ,
            ,
            ,
            "Answer N",
        ]
        "Marks": M
    }
}
```

每个问题都有一组答案，模型可以从中学习。每个答案通过以下步骤从自然语言变成输入向量。

1.  字数和句子数:计算字数和句子数。
2.  计算正确率:正确单词数与总单词数之比。
3.  停用字词删除:答案中所有停用字词都被删除。
4.  词性标注:过滤所有感叹词、介词、连词和代词。
5.  同义词包含:包括名词和形容词的同义词。
6.  频率分布:统计单词及其同义词的重复次数。

```
>>> vector = prepare_vector("I love to code. Code is addictive.")
>>> vector
{'word_count': 7, 'sentence_count': 2, 'correct_ratio': 1.0, 'clean_words': 3, 'love': 1, 'beloved': 1, 'code': 2, 'addictive': 1}
```

该向量的特征是单词计数、句子计数、正确率、单词(及其同义词)及其频率分布。为了适应最后一个特征，我们使用字典。一个示例输入看起来类似于上面的代码片段。

选择的机器学习模型之前是由 Analytics Vidhya 在 Medium 上发布的[，由我撰写](/analytics-vidhya/clustering-dissimilar-vectors-using-k-means-d4f09ddd3b0a)。它接受字典形式的输入，而不是 numpy 数组，以适应不同的特性，这里是单词。聚类的数量，即 K 的值等于数据集文件中的标记属性。每个问题将有一个单独的模型实例，以确保有歧义。

获得的聚类中心按照内容、呈现和正确性的优先级顺序用特征权重排序。这些可以在我们的向量中找到，如`clean_words`、`word_count`、`sentence_count`和`correct_ratio`。每个特性的权重如下。

```
feature_weights = {
    'clean_words': 0.5
    'word_count': 0.2,
    'sentence_count': 0.2,
    'correct_ratio': 0.1,
}
```

既然我们的模型已经训练好了，我们需要评估脚本。这些答案将通过以下格式的 JSON 文件提供给系统。

```
{
    "Question 1": "Answer 1",
    "Question 2": "Answer 2",
    ,
    ,
    ,
    "Question N": "Answer N"
}
```

该问题被用作获取预训练模型的关键。答案会经历数据集中的答案所经历的所有步骤。这些向量现在被传递给 predict()函数。利用返回的标签，识别匹配的聚类中心。匹配聚类在排序中心的位置给出了该特定答案的分数。

对于给定的每一个回答，从排序的中心中选择最佳聚类中心以及学生回答的向量。比较用于排序的加权特征。如果干净的话少了，内容就要增加。如果句子少了，表达就要改进。如果拼写错误较多，建议减少拼写错误。获得的输出是下面的格式。

```
{
    "question": {
        "answer": "The answer to the question",
        "marks_awarded": m,
        "max_marks": n,
        "feedback": "Necessary feedback to improve answer."
    }
}
```

因此，通过使用该系统，我们可以将评估的复杂性从类似 Map-Reduce 的系统降低到简单的桌面应用程序。

代码？这一切都发生在 https://github.com/AjayRajNelapudi/Automated-Answer-Grading。如果你喜欢，开始我的回购。