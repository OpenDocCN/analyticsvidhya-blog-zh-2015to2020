# 将 SQL 查询转换为 SQLALCHEMY ORM

> 原文：<https://medium.com/analytics-vidhya/translating-sql-queries-to-sqlalchemy-orm-a8603085762b?source=collection_archive---------7----------------------->

我用过很多 SQL。直到最近，我在为公司开发管道时才开始使用 SQLALCHEMY ORM。这是一个漂亮的 python 包，真的允许你模块化 SQL 查询。

本文只是展示如何将常用的 SQL 查询转换成 ORM 的一种方式。让我们创建一些基本结构。

```
$ python -V
Python 3.7.0$ ipythonIn [**1**]: **import** **sqlalchemy**

In [**2**]: sqlalchemy.__version__
Out[**2**]: '1.3.17'

In [**3**]: **from** **sqlalchemy** **import** create_engine

In [**4**]…
```