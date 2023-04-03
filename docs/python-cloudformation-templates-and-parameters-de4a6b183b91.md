# Python 云形成模板和参数

> 原文：<https://medium.com/analytics-vidhya/python-cloudformation-templates-and-parameters-de4a6b183b91?source=collection_archive---------16----------------------->

![](img/85143d23647616812ca00b1e85e19b50.png)

马克斯·尼尔森在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

我已经在 IT 行业工作了 10 年，主要是作为一名生产工程师，3-4 年前我成为了一名 devops 工程师。

最近，我一直在关注 AWS 和云在效率方面能给我们带来什么。这让我为一个外国客户实施了一个完整的 CI/CD 解决方案，从开发交付开始，到使用 Gitlab CI/CD 管道进行生产的基本环节。

如果你熟悉 AWS，你就不能忽视 Cloudformation(或者 Terraform，或者任何你想用来做一些基础设施如代码的工具)。

因此，在我们的 CI/CD 管道中，我们有如下的结构:
-每个 git 分支是一个环境
-每个环境有一个特定的参数文件
-一个公共的模板文件被所有的 env 使用

cloudformation 模板文件在这里描述了我们想要的基础设施:负载平衡器、DNS 条目、自动扩展组等等。

顾名思义，参数文件用于通过专用于特定环境的值来覆盖模板中的基本参数(例如，在 DEV 环境中，您的 VPC 可能与 PROD 环境中的不同)。

我们遇到的一个问题是，params 文件可能因环境而异，有时我们会丢失一些条目，这会导致管道故障。

为了避免这种情况，我们使用 python 脚本来比较这两个文件(我忘了说 param 文件是 JSON，而模板是 YAML ),如果它们之间有一些差异，就打印出来。

首先，我们对参数文件使用 json.load，对模板使用 yaml BaseLoader 来加载这两个文件。

json.load 和 yaml.load 函数将我们的文件转换为 dicts，因此我们可以通过使用各自的键提取想要的值来获得参数。

然后我们创建两个列表:paramListInParamFile 和 paramListInTemplateFile，我们从这两个文件中推送所有参数值。

最后，我们对列表进行排序，以便能够轻松地进行比较。

现在开始有意思了。

如果两个列表不相同，需要找出哪个键出现在列表 1 中，而不是列表 2 中。为此，我们用典型的 python 语法创建了一个结果列表

```
resultNotInParamFile = [x for x in paramListInTemplateFile if x not in paramListInParamFile]
```

这意味着:resulatNotInParamFile 将由“x”填充，只有当 x 不在 paramListInParamFile 中时，它才是 paramListInTemplateFile 的对象。

然后，如果我们的新列表的长度不为 0(即不为空)，我们从列表中取出所有的条目，并将它们转换为 string(参见 map 函数)并转换为一个新的 var，我们可以用它来打印结果(第 9 行和第 14 行)。

例如，这个脚本可以在部署之前运行，并触发您想要的任何东西(警报、自动纠正……)。

这是我的第一篇媒体文章(顺便说一下，在英语中，我是法国人:)

希望你喜欢。