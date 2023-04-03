# Gatsby 关于运行多个查询的提示(GraphQL 别名)

> 原文：<https://medium.com/analytics-vidhya/gatsby-tip-on-running-multiple-queries-graphql-aliases-dc978fe481da?source=collection_archive---------17----------------------->

![](img/7e3598e4764e77118afb46956fb26dec.png)

阿尔瓦罗·雷耶斯拍摄的照片

假设您想要在一个页面中获取基于参数或条件的特定数据，而这些数据不能使用一个查询来运行，因为您不能使用不同的条件或参数来查询相同的字段。一种方法是使用 [GraphQL 别名](https://graphql.org/learn/queries/#aliases)，您可以使用它将返回的数据集重命名为您想要的任何名称。

# 例子

```
export const query =…
```