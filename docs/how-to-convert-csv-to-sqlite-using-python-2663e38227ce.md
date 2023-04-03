# 如何转换。csv 到。使用 python 的 sqlite

> 原文：<https://medium.com/analytics-vidhya/how-to-convert-csv-to-sqlite-using-python-2663e38227ce?source=collection_archive---------8----------------------->

![](img/b55192e0a3524db7fd6f9d147a5e7b25.png)

约翰·施诺布里奇在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

***F*** 有些人非常熟悉 SQL，而不是使用 python 对数据运行条件，在大多数情况下，对数据运行 SQL 查询比 python 更高效，需要的行数也更少。

我们在 Python 中有一个名为“ [csv-to-sqlite](https://pypi.org/project/csv-to-sqlite/) 的包，我们可以使用 pip 导入模块。会改变信仰。csv 文件到。sqlite 文件。

```
pip install csv-to-sqlite
```

使用 pip 导入包安装后:

```
import csv_to_sqlite
```

导入包之后，转换为之前。sqlite 我们需要指定文件的类型(“your_csv_file.csv”)和数据的编码格式，以及数据的分隔符。默认分隔符是逗号('，')，我们可以根据 csv 文件中的数据格式进行更改。

```
options=csv_to_sqlite.CsvOptions(typing_style=”full”,encoding=”windows-1250")
csv_to_sqlite.write_csv("your_csv_file.csv", "output.sqlite", options)
```

执行完上面几行代码后，现在你的 csv 文件转换成了当前目录下的 sqlite。在本例中，它是用文件名“output.sqlite”创建的

现在我们有了。我们如何使用 Python 来运行 SQL 查询呢？？？？？？我们可以使用另一个名为“sqlite3”的包来实现。在导入这个包之后，我们需要与数据库建立连接，我们需要在数据库上执行 SQL 查询。

```
pip install sqlite3
```

在将 csv 转换成 sqlite 之后，有一点很重要。在 sqlite 文件中，数据库名称是 csv 文件的名称。这意味着我们在这里转换 your_csv_file . CSV-> output . sqlite。因此 SQLite 中的数据库表名是 your _ CSV _ file。

源代码:

```
import sqlite3
con=sqlite3.connect('output.sqlite')
filter_data=pd.read_sql_query("""SELECT * FROM your_csv_file""",con)
```

这里你的过滤数据是数据框。

使用[sqliteonline.com](https://sqliteonline.com/)有另一种处理 sqlite 文件的方法。我们可以上传我们的 sqlite 文件，我们可以在那里运行查询。