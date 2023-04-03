# Keras/TensorFlow 的自定义指标

> 原文：<https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05?source=collection_archive---------2----------------------->

![](img/4ceffc6ae0e6637a13395fdd2630e07c.png)

克里斯里德在 [Unsplash](https://unsplash.com/s/photos/code?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

最近，我发表了一篇关于二进制分类度量的文章，你可以在这里查看。这篇文章简要解释了最传统的指标，并介绍了不太有名的指标，如 NPV、特异性和 MCC。如果您不知道其中的一些指标，可以看看这篇文章。看书只要 7 分钟。我相信这对你会有用的。

在这篇文章中，我决定分享深度学习框架的这些指标的实现。它包括**召回率、精确度、特异性、阴性预测值(NPV)、f1-得分和马修斯相关系数(MCC)** 。在 Keras 或 TensorFlow v1/v2 中都可以使用。

# 代码

以下是所有指标的完整代码:

代码中几乎所有的指标都在前面提到的[文章](/analytics-vidhya/what-nobody-tells-you-about-binary-classification-metrics-4998574b668)中有描述。因此，你可以在那里找到详细的解释。

# 如何在 Keras 或 TensorFlow 中使用

如果你使用 Keras 或者 TensorFlow(尤其是 v2)，使用这样的度量是相当容易的。这里有一个例子:

```
model = ... # define you model as usualmodel.compile(
    optimizer="adam", # you can use any other optimizer
    loss='binary_crossentropy',
    metrics=[
        "accuracy",
        precision,
        recall,
        f1,
        fbeta,
        specificity,
        negative_predictive_value,
        matthews_correlation_coefficient,
        equal_error_rate
    ]
)model.fit(...) # train your model
```

如您所见，您可以一次计算所有的定制指标。请记住:

*   **由于它们是二进制分类指标，您只能在二进制分类问题中使用它们**。也许你会有一些多类或回归问题的结果，但它们将是不正确的。
*   **它们只能用作衡量标准。**意思是你不能把它们当做损失。事实上，你的损失一定总是“ **binary_crossentropy** ”，因为这是一个二进制分类问题。

# 最后的话

我希望你喜欢这篇文章。如果这对你也有帮助，请给点掌声👏👏。关注我在媒体上更多这样的职位。您也可以在以下位置查看我的作品:

*   [Github](https://github.com/arnaldog12)
*   [领英](https://www.linkedin.com/in/arnaldo-gualberto/)
*   [个人网站](http://www.arnaldogualberto.com/)

![](img/11dbe24ef472fd228b1e66d4b178f76e.png)