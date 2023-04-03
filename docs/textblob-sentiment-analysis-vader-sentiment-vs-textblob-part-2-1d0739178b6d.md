# 文本分析(VADER 情绪 vs 文本分析) :第 2 部分

> 原文：<https://medium.com/analytics-vidhya/textblob-sentiment-analysis-vader-sentiment-vs-textblob-part-2-1d0739178b6d?source=collection_archive---------15----------------------->

你已经学习了情感分析的重要性，使用 Python 进行情感分析，以及 VADER 情感包。现在，让我们看看用于情感分析的 TextBlob 包。

如果您还没有看过第 1 部分，请点击[查看。](/@jeffsabarman/sentiment-analysis-vader-sentiment-vs-textblob-part-1-a8ccd2d610de)有就去吧！

![](img/2f38c99d47d124a23e21065a8e2f47f9.png)

盒子里的表情符号(来源:Pexels.com)

我们已经涵盖了步骤 1、2 和 3。第一步包括软件包和 IDE 的安装。第二步包括导入必要的包。第三步是 VADER 情绪计划。现在，让我们继续我们的旅程到第 4 步！

# 步骤 4:文本 Blob 情感分析

TextBlob 有很多功能。TextBlob 可以做标记化、N-grams、词性标注等等。你可以在这里获得[的整体信息。在本文中，我们将讨论 TextBlob 中的情感分析部分。](https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/)

还记得我们用 VADER 情感做情感分析的感觉吗？如果不是，下面是我们要对比的句子:

```
sentence1 = "I love this movie so much!"
sentence2 = "I hate this move so much!"
```

现在，让我们做文本情感分析。首先，我们必须这样做:

```
text1 = TextBlob(sentence1)
text2 = TextBlob(sentence2)
```

每个句子的文本块语法被用来获得句子的情感。之后，我们可以得到句子的情感。

```
score1 = text1.sentiment
score2 = text2.sentiment
```

得分 1 和得分 2 的结果如下:

```
This is the score for the first statement : Sentiment(polarity=0.375, subjectivity=0.4)This is the score for the second statement : Sentiment(polarity=-0.275, subjectivity=0.55)
```

你可以看到 TextBlob 给了我们一个带有极性和主观性的分数。极性的值介于-1 和 1 之间。如果极性值是负值，则该句子具有负面情绪，反之亦然。所以，极性衡量情绪。值越接近 1，句子越积极，反之亦然。

主观性的值介于 0 和 1 之间。主观性就像名字一样衡量句子中的主观性。值越小，句子越有可能是一种观点。值越大，句子最有可能是事实信息。

你可以得到每个句子的最终情绪:

```
if score1.polarity >= 0:
        print("The first sentence is  a positive sentiment :)")
elif score1.polarity < 0 : #or else
        print("The first sentence is a negative sentiment :(")
if score2.polarity >= 0:
        print("The second sentence is  a positive sentiment :)")
elif score2.polarity < 0 : #or else
        print("The second sentence is a negative sentiment :(")
```

上面代码的结果是:

```
The first sentence is  a positive sentiment :)The second sentence is a negative sentiment :(
```

如你所见，第一句是积极的情绪，第二句是消极的情绪。但是正如你所看到的，你也必须看看主观性，才能知道这个句子是观点还是事实信息。有时，观点和事实信息对于确定句子的实际情感是有用的。

如果我们比较 VADER 情绪，你可以看到 VADER 情绪可以给你每个句子的消极，积极和中性的分数。它计算句子中的所有单词，并把它归入那三个类别。之后，这个句子会有一个复合分数(总分数)。TextBlob 给我们的只是一个极性和主观性。正如你所看到的，每个情绪分析包都有优点和缺点。

这就是为什么，在我关于 IMDB review 的 [web 爬行和 web 抓取的文章中，我使用了这两个包。](/analytics-vidhya/movie-recommendation-from-imdb-reviews-without-actually-read-the-reviews-fe8865a70bd5)

这个系列将会有一个关于情感分析的第三部分(VADER 情感与文本分析)。在第 3 部分中，我们将使用来自 [Kaggle](https://www.kaggle.com/datasets) 的 IMDB review 来比较软件包的准确性。

一定要关注我，这样以后就不会错过我的文章了。如果你对任何步骤有任何麻烦或困惑，请在 instagram 上联系我或查看我的 [github](https://github.com/jeffsabarman) 。

如果你喜欢这类内容，请告诉我！你可以在 instagram @jeffsabarman 和 [linkedin](https://www.linkedin.com/in/jeffrey-sabarman-009065141/) 上找到我。