# 在 Tableau 中可视化数据块仪表板

> 原文：<https://medium.com/analytics-vidhya/visualize-databricks-dashboards-in-tableau-eae01b5c7219?source=collection_archive---------6----------------------->

**在这篇文章中，我将描述设置一个笔记本的步骤，该笔记本将数据块仪表板导出为 HTML 文件，并将其上传到为静态网站托管** **配置的 S3 桶** [**。在 Tableau 中，我们将创建一个仪表板，嵌入文件所在的 URL。**](https://docs.aws.amazon.com/AmazonS3/latest/dev/WebsiteHosting.html)

![](img/20df5f51dc8aa4eda0dd3bfaddd4b300.png)

[https://databricks.com/partners/tableau](https://databricks.com/partners/tableau)

笔记本和数据可视化工具是企业数据框架的重要组成部分。笔记本主要是…