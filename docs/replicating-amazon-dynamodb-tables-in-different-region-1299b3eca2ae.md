# 在不同地区复制 Amazon DynamoDB 表

> 原文：<https://medium.com/analytics-vidhya/replicating-amazon-dynamodb-tables-in-different-region-1299b3eca2ae?source=collection_archive---------1----------------------->

![](img/db0e6112834c39737a639fd18f9ce389.png)

使用 AWS DynamoDB 的开发人员经常会在不同的区域复制一个或多个表。这可以使用 AWS S3 和 AWS 数据管道来完成，如这里的[所解释的](https://aws.amazon.com/blogs/aws/cross-region-import-and-export-of-dynamodb-tables/)。

但是对于那些不喜欢做*点击*、*点击*、*点击*的铁杆命令行粉丝来说(和我一样)。下面是一个使用 boto3 的简单 python 脚本，它将使用相同的键模式、属性定义在不同的区域复制所需的表，并将数据复制到其中。

## 脚本的工作原理:

*   扫描现有表中的模式、属性和数据。
*   在新区域中创建一个具有相同模式和属性定义的新表。
*   通常，DynamoDB 需要几秒钟来创建表。这由***wait _ until _ exists()***处理，它每 20 秒轮询一次，直到达到成功状态。
*   重新加载以更新表资源的属性，从而获得实际的表状态。
*   新表格激活后，将从现有表格扫描的所有项目写入其中。

[https://gist . github . com/Dineshkarthik/d 0944 c 45 b 06726 a 327 a 9536 a 33 dab D2](https://gist.github.com/Dineshkarthik/d0944c45b06726a327a9536a33dabdd2)

```
*python dynamodb_replicate_table.py -t my-table -r* eu-west-1 *-nr us-east-2*
```

上面将把 eu-west-1 中名为“my-table”的表复制到 us-east-2 中。