# 将配置单元表导出到 CSV 文件的最佳方式

> 原文：<https://medium.com/analytics-vidhya/best-way-to-export-hive-table-to-csv-file-326063f0f229?source=collection_archive---------0----------------------->

## 这篇文章是为了解释不同的选项可用于导出配置单元表(ORC，拼花或文本)到 CSV 文件。

**预期输出:**带逗号分隔符和标题的 CSV 文件

![](img/c95f2c019251333332d66ade02ccdc6c.png)

**方法一:**

```
hive -e 'select * from table_orc_data;' | sed 's/[[:space:]]\+/,/g' > ~/output.csv
```