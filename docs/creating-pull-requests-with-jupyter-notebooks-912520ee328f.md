# 使用 Jupyter 笔记本创建拉取请求

> 原文：<https://medium.com/analytics-vidhya/creating-pull-requests-with-jupyter-notebooks-912520ee328f?source=collection_archive---------6----------------------->

你有没有想过，如果一个人忘记标注图表的 x 轴，你会如何给这个人反馈？

![](img/0d960f978abc3c6112054941a206a40a.png)

不熟悉 Jupyter 笔记本？查看数据营的[教程](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)

当我开始做后端开发的时候，那还是蛮荒的西部。我可以在没有任何人查看我的代码的情况下进行部署，只有当一个客户打电话告诉我某个东西已经崩溃时，我才知道出了什么问题。有一天，通过一次活动的对话，我发现了拉式请求(PRs)的存在！有了它们，其他开发人员可以看到我的代码，并给我各种反馈，比如我如何改进我的代码，他们各自的实现有什么好的做法。处理拉式请求有很多好处，但是我发现最令人惊奇的是它们支持的知识共享。关于使用 bitbucket 创建拉请求的更多细节，请查看这个[链接](https://www.atlassian.com/git/tutorials/making-a-pull-request)。

当我成为一名数据科学家后，一切都变了。我开始使用 Jupyter 笔记本，它对我的研究和图表的可视化以及我的代码的结果非常好，但对制作 PRs 来说却很糟糕。Github 渲染笔记本，但无法评论。在 PR 模式下，出现的是一个不容易阅读的 JSON。

![](img/2e398de96658170752a65e7648c257cd.png)

为了解决这个问题，我们搜索了一些社区，尤其是我非常欣赏的一个项目的社区:“ [Serenata de Amor](https://serenata.ai/) ”(爱情小夜曲)。在那里，我们找到了一个相对有效的方法:生成一个. py 和一个. ipynb。要在每次保存笔记本时自动完成这项工作，只需将以下代码添加到文件中:`~/.jupyter/jupyter_notebook_config.py`:

存在的方式很简单，但也有不同的方式。

这将自动使创建的文件数量翻倍！但是，它允许您查看笔记本并在上发表评论。py 文件，在单元格中，有人在其中修改它是有意义的。

![](img/2f9f6a7f6083a71c5be6a45b0e37f833.png)

值得一提的是，除了发出拉请求之外，还有其他选择，特别是如果您正在使用开源代码的话。其中一个选择是 [reviewNB](https://www.reviewnb.com/) ，它允许将评论直接留在笔记本的单元格中，但这种解决方案不再对私有存储库免费，并且只适用于 Github(这对 GitLab 和 BitBucket 用户来说是不幸的)。您也可以使用 [nbviewer](https://nbviewer.jupyter.org/) 对笔记本电脑进行测试。

![](img/da29da8ae2ee16966cb68e706f957321.png)

当你的 PR 在 888 条评论后被批准时的心情。

我们遵循的另一个好的实践是使用稍微修改过的版本的 Cookiecutter 数据科学来组织我们的项目。这样，我们遵循“*笔记本用于探索和交流*”的规则——这样，数据提取代码、特性工程和调优模型保存在其他地方，而笔记本主要用于 EDAs ( *探索性数据分析*)和评估。这极大地方便了这些代码的版本化和执行。

Cookiecutter 的文件夹结构使得团队中的任何人都可以工作，并为 Creditas 的所有数据科学家的研究做出贡献。

有兴趣与我们合作吗？我们一直在寻找热爱技术的人加入我们的团队！你可以在这里查看我们的[职位空缺](https://vagas.creditas.com.br/)。