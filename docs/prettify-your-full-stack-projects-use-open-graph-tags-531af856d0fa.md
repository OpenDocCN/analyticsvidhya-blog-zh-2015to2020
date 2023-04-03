# 美化你的全栈项目:使用开放的图形标签！

> 原文：<https://medium.com/analytics-vidhya/prettify-your-full-stack-projects-use-open-graph-tags-531af856d0fa?source=collection_archive---------21----------------------->

![](img/042a205c9eaab453768455fbddc92030.png)

[伊维·s .](https://unsplash.com/@evieshaffer?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

嗨伙计们！

对你们来说，这将是一篇更小更快的文章。

你有没有想过如何制作和利用更好的社交媒体链接？比如当你在脸书、Twitter 或 LinkedIn 上分享东西时，可以控制弹出的预览框？

那是[脸书的开放图协议，你可以在这里看到解释](https://ogp.me/)。

现在，您可能想知道这与数据科学有什么关系，因为我通常专注于此！作为数据科学家，你应该记住的一件事是你的终端用户。无论是可视化还是模型结果，你都必须让“客户”容易理解。只要你按时交付结果，有商业头脑的人可能不会太担心你需要多少次模型迭代，如果你的模型不容易验证或部署，软件工程师或其他数据科学家可能会与你的模型斗争。

因此，如果您正在制作或已经制作了一个完整的项目，您希望它能够很好地呈现，并为用户带来愉快的体验。这就是我失败的地方。

![](img/436e0dcf7cf241af62e6b1dc330512d9.png)

由[凯利·西克玛](https://unsplash.com/@kellysikkema?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

我不明白 Open Graph 是如何工作的，而且 [MeaLeon(我的由机器学习驱动的食谱推荐器)](http://mealeon.herokuapp.com/)每当我分享它的链接时，完全依赖于我控制之外的东西。我注意到了这一点，因为我的 LinkedIn 链接只显示了损坏的缩略图，这让我很恼火，我想不出为什么它不起作用。

当我进行初步搜索以找出原因时，人们提到了开放图标签。然而，我找到的关于它的帖子似乎忽略了脸书文档中提到的一小部分:你至少需要 4 个标签才能让它工作！我只放了一个(`og:image`)，似乎导致标签没有被使用。

您需要在 HTML 的中包含以下内容:

```
<meta property=”og:title” content=”NAME OF YOUR CONTENT” /><meta property=”og:type”  content=”WHAT IS THE CONTENT” /><meta property=”og:url”   content=”URL OF THE SITE” /><meta property=”og:image” content=”URL OF THE IMAGE YOU WANT DISPLAYED” />
```

ogp.me 站点列出了您可以添加的所有额外属性，但这 4 个属性是您需要开始的全部内容。如果你想看看元标签是否有效，检查你在这些地方的链接，从[脸书](https://developers.facebook.com/tools/debug/)和 [LinkedIn](https://www.linkedin.com/post-inspector/) ！

我从 YouTube 上的一些 T4 营销人员那里学到了最后一招。

总之，这一周到此为止。让你的社交媒体帖子变得更漂亮，让人们看到，坚持下去，经常洗手，保持积极的心态！