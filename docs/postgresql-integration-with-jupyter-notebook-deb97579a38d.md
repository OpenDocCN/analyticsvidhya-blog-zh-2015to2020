# PostgreSQL 与 Jupyter 笔记本的集成

> 原文：<https://medium.com/analytics-vidhya/postgresql-integration-with-jupyter-notebook-deb97579a38d?source=collection_archive---------0----------------------->

*获得更加无缝的数据提取体验。*

![](img/c13aa872c2992d3497e4ddae5be32034.png)

由[paweczerwiński](https://unsplash.com/@pawel_czerwinski?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

Jupyter Notebook 是一款很棒的软件，可以记录数据可视化和分析过程中采取的步骤。它支持 Python 代码、markdown、HTML，多亏了几个库，还有 PostgreSQL！这允许您在笔记本中编写 SQL 代码并进行更改，或者从那里查询数据库。您还可以将任何查询转换成 pandas 数据框架，以便直接进行分析。它让我更喜欢使用 SQL 和 Jupyter 笔记本！

## 先决条件:

*   Jupyter 笔记本
*   PostgreSQL 和一个数据库([示例数据库&如何加载](http://www.postgresqltutorial.com/postgresql-sample-database/))

## 要安装的库:

*   [ipython-sql](https://github.com/catherinedevlin/ipython-sql)
*   sqlalchemy
*   python 数据库 API (DBAPI)库

前两个库可以使用 pip install 命令安装:

```
pip install ipython-sql
pip install sqlalchemy
```

第三个库取决于您选择使用的 SQL 软件。对于 PostgreSQL，您可以使用 [psycopg2](https://pypi.org/project/psycopg2/) :

```
pip install psycopg2
```

不同的 DBAPIs 都记录在 sqlalchemy 的网站上，比如 MySQL、Oracle 和 SQLite(感谢 Dave Campbell)。

## ipython-sql

由 Github 上的 [catherinedevlin 创建，它支持使用包含`%`和`%%`的 SQL 魔法函数，允许您在 Jupyter Notebook 中编写 SQL 风格的代码。](https://github.com/catherinedevlin/ipython-sql)

## sqlalchemy

sqlalchemy 最初由 Michael Bayer 创作，被吹捧为 Python 的“SQL 工具包和对象关系映射器”。出于本教程的目的，它将主要用于将 SQL 查询存储到 pandas 数据帧中。

现在我们已经安装了我们的库，让我们继续 Jupyter 笔记本！

# 入门指南

要加载 ipython-sql，请使用以下神奇的命令:

```
%load_ext sql
```

接下来，我们将只需要 sqlalchemy 中的`create_engine()`函数，所以让我们用下面一行来导入它:

```
from sqlalchemy import create_engine
```

一旦我们奠定了基础，我们现在就可以连接到 PostgreSQL 数据库了！

# 连接到 PostgreSQL 数据库

要将 ipython-sql 连接到您的数据库，请使用以下格式:

```
# Format
%sql dialect+driver://username:password@host:port/database# Example format
%sql postgresql://postgres:password123@localhost/dvdrental
```

*   在这种情况下，`dialect+driver`就是`postgresql`，但是在这里可以随意使用不同的数据库软件。
*   `username:password`是您替换用户名和密码的地方。
*   `host`通常只是 localhost。
*   在我的经验中，`port`不需要被指定。
*   `database`是要连接的数据库的名称。

为了查看它是否成功连接，Jupyter 笔记本将打印以下内容:

```
'Connected: username@database'
```

为了将 sqlalchemy 连接到数据库，格式基本相同，但是我们将使用`create_engine()`函数创建一个名为`engine`的新对象:

```
# Format
engine = create_engine('dialect+driver://username:password@host:port/database')# Example format
engine = create_engine('postgresql://postgres:password123@localhost/dvdrental')
```

在大多数情况下，当连接到 ipython-sql 时，您可以复制并粘贴您在`%sql` magic 命令后编写的内容，并将其封装在引号中作为`create_engine()`函数中的参数。

只要您在输出中看不到错误，您就应该连接到数据库了！

# 在 Jupyter 笔记本中编写 SQL 命令

要启用数据库查询和其他命令，调用神奇的命令`%%sql`并在其后添加您的 SQL 代码。SQL 代码应该在它自己的块中，与 Python 代码分开(感谢 Daniel Upton 指出这一点！).下面是查询 inventory 表中前三行和所有列的示例代码:

![](img/d9322c41cd7fb7dd02de19623d30e7d3.png)

如果您想将查询存储在 pandas 数据框架中，这就是 sqlalchemy 的用武之地。使用命令`pd.read_sql()`创建一个 dataframe 对象。它需要两个参数:

1.  用引号括起来的 SQL 查询。
2.  您之前使用`create_engine()`函数创建的`engine`对象。

以下是如何使用上图中的库存表查询 SQL 代码实现数据帧`inventory_dataframe`的示例。

![](img/4af237bb2935ff097a43482b84aaeb4c.png)

一旦在数据框架中有了查询，就可以放心地继续进行数据分析了！