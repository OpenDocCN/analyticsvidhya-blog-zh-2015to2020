# python 和 Django 开发人员的便利资源和工具

> 原文：<https://medium.com/analytics-vidhya/handy-resources-and-tools-for-python-and-django-developers-348e8959b819?source=collection_archive---------8----------------------->

![](img/621589dae0758c3f0ab73e538a132334.png)

费萨尔·米在 [Unsplash](https://unsplash.com/s/photos/django?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

python 和 Django 开发人员的便利资源和工具

我使用 Django 开发软件已经有一段时间了，这些年来，我已经开始依赖一些资源和库。它们已经过测试，可以轻松扩展以满足我的需求。

我写了另一篇文章[强调初级开发人员的资源](https://lewiskori.com/blog/list-of-resources-for-junior-developers/)。有兴趣可以去看看。

我将从一些技术内容创作者开始，他们对我在 Django 开发中的成功有着不可估量的作用。

# 内容创建者

# 简单比复杂好

这是一个由熊伟·弗雷塔斯经营的博客，提供 Django 的技术知识和技巧。我强烈推荐。

# 真正的 Python

又一个长期运行的博客。每个 python 开发人员在为他们的项目做研究时，至少会遇到一两次这个平台。

# 绿色代码

这是一个由 Marvin Kweyu 经营的博客，他是一名肯尼亚软件开发者，自称是 Djangonaout 人。这些文章研究得很好，写得很清楚。

# 科里·舍费尔

youtube 频道有大量的资源。这些不仅限于 Django 开发，还包括一般的 python 编程。他的激情和奉献精神可以在每一个视频中感受到，你一定会从那里学到很多东西。

# 新波士顿

这是我做网站开发的第一站，Bucky 对软件世界的贡献不可低估。很高兴看到[他在长期中断后又回到了内容创作](https://www.youtube.com/watch?v=D-3i1g5YFik)。

# 谈谈 python FM

talkpython.fm 是一个由 Michael Kennedy 主持的播客，他们在这里讨论 python-dev 世界中的各种问题。他们在 [Django 发展最佳实践](https://talkpython.fm/episodes/show/277/10-tips-every-django-developer-should-know)中的一集特别有用，令人大开眼界。

# 在线社区

尽管内容创建者提供了大量信息，但当您需要回答一些紧迫的问题时，他们可能并不总是能找到您。这就是在线社区填补空白的地方。这些都是志同道合的人的平台，你可以接触到他们，在大多数情况下，他们会回答你的问题或帮助你以这样或那样的方式成长。

# 开发到 python 标签

[dev.to](https://dev.to/) 是一个由软件开发人员和对软件开发感兴趣的人组成的社区。它以标签的形式组织它们的内容。我最喜欢的一个是 [#python](https://dev.to/t/python) 标签，来自世界各地的人们经常在这里发布他们的内容。无论是问题还是技术博客帖子/播客。

# Indiehackers django 集团

一个独立的黑客根据平台是:

1.  人建立一个在线项目，可以产生收入。
2.  寻求经济独立、创作自由和按照自己的时间表工作的人。

indiehackers.com 正是这样一个由在线用户组成的社区，他们寻求建立在线项目来创造收入，并使他们获得财务自由。Django 是最大的软件开发框架之一，是实现这些梦想的推动者。所以这个平台应该有一个[专门的 Django 小组](https://www.indiehackers.com/group/django)来服务那些相信这个框架将他们的梦想变成现实的成百上千的企业家，这是再合适不过的了。

# PythonKe 电报集团

作为肯尼亚内罗毕的一名软件开发人员， [telegram group](https://t.me/pythonKE) 让我了解并关注影响我所在地区开发人员的问题，并为我提供一个可以快速获得问题答案的地方。

# 软件库

# Djangorestframework

Django REST 框架是一个强大而灵活的工具包，用于使用 Django 构建 Web APIs。Django 本身支持 JSON 序列化，但是 DRF 让这个过程变得如此简单。API 开发方法使构建解决方案变得容易，使您能够使用 react、angular 和 Vue 等现代 javascript 框架创建前端。可能性是无限的。

[Django-cors-headers](https://pypi.org/project/django-cors-headers/)Django 应用，将跨源资源共享(CORS)头添加到响应中。这允许来自其他来源的对 Django 应用程序的浏览器内请求。添加 CORS 标头允许您的资源在其他域上被访问。但是，在添加标题之前理解其含义是很重要的，因为您可能会无意中向他人公开您站点的私人数据。cors-headers-site 提供了许多资源来帮助您理解这些含义。

# 乔泽

[Djoser](https://djoser.readthedocs.io/en/latest/introduction.html) 库提供了一组 Django Rest 框架视图和端点来处理基本操作，比如注册、登录、注销、密码重置和帐户激活。您可以自己构建这些功能，但这已经包含了您最终可能要构建的大部分功能。

我的上一篇文章强调了使用 jwts 的用户认证和授权。全部由 Djoser 驱动。

# Django 仓库

Django-storages 是 Django 定制存储后端的集合。该库允许您配置 Django 将静态和媒体文件存储到各种平台，如 amazon s3、 [digital ocean](https://m.do.co/c/2282403be01f) 、google cloud 和 Dropbox。

# Django 频道和 djangochannelsrestframework

[djangochannelsrestframework](https://pypi.org/project/djangochannelsrestframework/)提供了一个类似 DRF 的接口，用于构建通道-v2 WebSocket 消费者。该库利用了 [djangorestframework](https://lewiskori.com/blog/handy-resources-and-tools-for-python-and-django-developers/#djangorestframework) 和 [django-channels](https://channels.readthedocs.io/en/stable/) 的能力来提供两者的无缝集成。

其他著名的图书馆有:

1.  姜戈-阿劳斯
2.  [姜戈-过滤器](https://django-filter.readthedocs.io/en/stable/)
3.  [python 解耦](https://pypi.org/project/python-decouple/)

# 结论

我就说这些。如果你有任何额外的资源，请在评论中提出。我很好奇你用什么工具。

## 开放合作

我最近在自己的网站上制作了一个合作页面。心中有一个有趣的项目或想填补一个兼职的角色？您现在可以直接从我的网站上[预订与我的会话](https://lewiskori.com/collaborate)。

*最初发表于*[*https://lewiskori.com*](https://lewiskori.com/blog/handy-resources-and-tools-for-python-and-django-developers/)*。*