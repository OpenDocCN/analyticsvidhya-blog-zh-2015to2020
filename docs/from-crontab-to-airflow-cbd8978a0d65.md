# 将 Crontab 迁移到 Airflow

> 原文：<https://medium.com/analytics-vidhya/from-crontab-to-airflow-cbd8978a0d65?source=collection_archive---------8----------------------->

![](img/a08502b22b18832a5015182eb506b71c.png)

图片来源:[https://en.wikipedia.org/wiki/Apache_Airflow](https://en.wikipedia.org/wiki/Apache_Airflow)&[https://www.team-azerty.com/](https://www.team-azerty.com/2016/05/20/actu-433-les-taches-cron-sont-de-retour.html)

作为一名开发人员，很有可能需要设置许多 cronjobs 来 ***执行常规任务*** 。但是，随着时间的推移，Cronjobs 会越来越多。此时此刻， ***为你的团队选择一款合适的 WMS(工作流管理系统)*** 是很重要且不容忽视的。

***气流*** 是管理你的 cronjobs 的 WMS 工具之一。这将节省您的时间来检查任务的条件，因为气流…