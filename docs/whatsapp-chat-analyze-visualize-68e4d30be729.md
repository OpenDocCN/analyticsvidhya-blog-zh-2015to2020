# WhatsApp 聊天📱—分析🔍，可视化📊

> 原文：<https://medium.com/analytics-vidhya/whatsapp-chat-analyze-visualize-68e4d30be729?source=collection_archive---------2----------------------->

> WhatsApp 是当今世界上最受欢迎的即时通讯应用，在全球拥有超过 2B 的用户。每天发送超过 650 万条消息。

![](img/a1761ae180f88b7afb9b29bbb9ab9467.png)

你好。我在用 **WhatsApp** 。

在这篇文章中，我将向你展示任何 WhatsApp 聊天工具都可以进行的有趣的分析和可视化。

此练习的完整脚本可从以下位置获得:

[](https://github.com/SinghalHarsh/WhatsApp-Chat-Analysis/blob/master/whatsapp_chat_analysis.ipynb) [## SinghalHarsh/WhatsApp-聊天-分析

### permalink dissolve GitHub 是超过 5000 万开发人员的家园，他们一起工作来托管和审查代码，管理…

github.com](https://github.com/SinghalHarsh/WhatsApp-Chat-Analysis/blob/master/whatsapp_chat_analysis.ipynb) 

## 分析我们自己的数据是如此有趣！相信我！

# 获取 WhatsApp 聊天

```
WhatsApp has a functionality that enables you to download the conversation logs of individual and group chats.
```

> **iPhone** :打开聊天|点击姓名|向下滚动|导出聊天➞ 文本文件
> 
> **安卓**:打开聊天|轻点更多选项*|更多|导出聊天➞ 文本文件*

# *数据准备*

```
*Chat data is in a semi-structured format. Therefore, we need to convert it into a structured format to enable us to analyze and visualize data in a more interpretable way.Each line in a text file follows a specific format:
**[date, time] Author: message**
Using RegEx, we will parse the text file and convert it into a pandas dataframe.*
```

*[**代码**](https://github.com/SinghalHarsh/WhatsApp-Chat-Analysis/blob/master/helper.py)*

*![](img/0634e90898c2a2f2df7206d2edb924c0.png)**![](img/dae7ec7e4e873e9a511e0c4a2510ecfe.png)*

*文本文件➞熊猫数据帧(使用**正则表达式***

# *探索性数据分析*

> ***首先，一些基本的统计数据***

```
*How many messages have been exchanged?
How many authors are there?
What is the average number of messages exchanged every day?*
```

*![](img/02d2b105030a00d9e77d35234e9f1165.png)*

> ***分析 1:日期时间***

```
*When was the group most active?
Which day of the week, part of the day, an hour of the day has the most number of messages exchanged?*
```

*![](img/d358f3498f6ac4248fe6284ee85a3621.png)**![](img/bee9d665f969e1d943b9a81c5097281b.png)**![](img/361206bb35e9665fd090c032d5001cf6.png)*

***右图**:周末🕺*

*![](img/3beac22bd2e69f647f63c80a0b7b2e07.png)**![](img/9e4445857a7a8bdf5a8256210ba00671.png)*

***左**:猫头鹰🦉👻| **右**:周五晚上🙈*

> ***分析二:作者***

```
*Who is the most talkative?
Whose messages are decreasing with time?
Who sends long messages?*
```

*![](img/9d21327e7f62a14a487648f2d0577632.png)**![](img/2d1898fef60dd483b63df97289782171.png)**![](img/79d4e59763d81de6ea2c3d09491766b4.png)**![](img/25f73c737089086215f1ef5499ee4732.png)**![](img/2ab092d52c899c6ce5a1903485472ae5.png)**![](img/a340fa364d80a9e35bba008f8537ea05.png)*

***左**:全局| **中心**:作者 1 | **右**:作者 13*

> ***分析 3:消息***

```
*What are the most commonly used words in the messages (overall, author-wise)?*
```

*![](img/e58d7c8b982ecbde50f047f178e3ff7d.png)**![](img/f049eadd7db230ad40ac388411c855b3.png)**![](img/54cd94e591798fd2508b0d181a7bfbba.png)*

***左**:总体| **中心**:作者 2 | **右**:作者 5*

> ***分析四:表情符号***

```
*What are the most commonly used emojis (overall, author-wise)?
What is the emoji-to-message ratio for the author?*
```

*![](img/8e16cdeb86f1f7c974a9139c3ed72c5f.png)**![](img/e12c22701571b9cc02ffff0e6023124e.png)*

*‎*

*![](img/0dc80fc1d5e1525275ff7ac49b32e508.png)**![](img/b5c9d3b1d5fe3ba3fcb37d3407f07858.png)**![](img/9f777d8241433eebcb67d3ea22bbd956.png)*

***左**:全局| **中心**:作者 1 | **右**:作者 6*

> ***分析五:主题***

```
*What is the most common subject of the group?*
```

*![](img/dee06f8fe1e074048c706b75e6bb0f42.png)**![](img/11d705a2765f95815ed1466d7c9053a3.png)*

*13 个质数！！*

> ***分析 6:活跃性***

```
*How many days the group was silent?
Who is the most active author?*
```

*![](img/9d752e11916ca980cd4c8064607ce3b4.png)*

*‎*

*![](img/759106941b04e4207816f920471cc98b.png)**![](img/85833acc0068ba6c0ecc106db058dfe3.png)*

*‎*

*![](img/f4a9b20741953a743477500c64233e26.png)**![](img/157b0e76a089e2891b45e051a45c50b1.png)*

> ***分析 7:删除的消息***

```
*Which author has deleted the most number of messages?*
```

*![](img/8f431ed3d9a9b0d093c99dc9674fdcbe.png)*

> ***分析 8:互动***

```
*Whom the author has replied the most?
Who are the top responders to that author?*
```

*![](img/3edb2eaa63dcf8cddf5bb57383bfaafe.png)*

> ***分析 9:感悟***

```
*Who is the most positive author?
When was the group or the author most happy?*
```

*![](img/732271638b1de45f50af0a080bbb8e6a.png)**![](img/49d059a03c506c8c3abc4e7e50f2b45b.png)*

*左:统计|右:示例*

*![](img/9509b4c715887e41c67323f4133206e4.png)**![](img/85eb790a4dd1c4e68314b011e5d471e4.png)**![](img/a179f650f7520decf51b8ee6be9478dc.png)**![](img/ea08e1014e93b0c8ac3d03a0ee3ad94d.png)*

***左**:总体| **中心**:作者 2 | **右**:作者 10*

# *感谢阅读这篇文章！如果你有任何问题，欢迎在下面留言。*

# *参考资料:*

> ***RegEx:**https://regexr.com/*
> 
> *[https://towards data science . com/build-your-own-whatsapp-chat-analyzer-9590 acca 9014](https://towardsdatascience.com/build-your-own-whatsapp-chat-analyzer-9590acca9014)*
> 
> *https://github.com/PetengDedet/WhatsApp-Analyzer*
> 
> *[https://level up . git connected . com/text-and-opinion-analysis-of-whatsapp-messages-1 eebc 983 a 58](https://levelup.gitconnected.com/text-and-sentiment-analysis-of-whatsapp-messages-1eebc983a58)*