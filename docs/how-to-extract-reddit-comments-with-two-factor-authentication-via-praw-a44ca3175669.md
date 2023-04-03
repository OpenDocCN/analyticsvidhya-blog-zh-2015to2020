# 如何通过 PRAW 用双因素认证提取 Reddit 评论

> 原文：<https://medium.com/analytics-vidhya/how-to-extract-reddit-comments-with-two-factor-authentication-via-praw-a44ca3175669?source=collection_archive---------16----------------------->

![](img/8f170796471cedf36903bd8eea37ecb0.png)

斯蒂芬·保罗在 [Unsplash](https://unsplash.com/s/photos/scenic-valleys?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

我已经两年多没有在任何 [Python 项目中使用 Reddit/PRAW 了](https://rareloot.medium.com/basics-of-data-extraction-of-reddit-threads-using-python-c96854c41344)，最近我在使用 [Python 的 Reddit API 包装器——PRAW](https://praw.readthedocs.io/en/latest/)提取评论时遇到了一个小问题。

如果您在 Reddit 帐户上启用了 2FA，那么访问 Reddit API 超过一小时需要您定期(每小时)验证您的访问权限…