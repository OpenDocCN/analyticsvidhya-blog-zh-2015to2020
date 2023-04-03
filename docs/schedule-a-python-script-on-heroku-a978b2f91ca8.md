# 在 Heroku 中计划 Python 脚本

> 原文：<https://medium.com/analytics-vidhya/schedule-a-python-script-on-heroku-a978b2f91ca8?source=collection_archive---------0----------------------->

![](img/e370e3bc1ebabcc712da620f1489205e.png)

快速故事..！在这次新冠肺炎疫情期间，我想收集新冠肺炎病例的日常数据进行分析。我能找到的最有希望的资源之一是 https://www.worldometers.info/coronavirus。但是我找不到那个特定站点的任何 API 端点，最后我用我自己的 python 脚本(使用漂亮的 Soup)来抓取数据并放入我的数据库。下一步是将该脚本托管为…