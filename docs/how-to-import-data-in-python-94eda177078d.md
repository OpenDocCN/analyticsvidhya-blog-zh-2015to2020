# 如何在 python 中导入数据？

> 原文：<https://medium.com/analytics-vidhya/how-to-import-data-in-python-94eda177078d?source=collection_archive---------16----------------------->

在每一种编程语言中，都有一些非常简单的非常基本的东西，但有时，我们只是在正确的时刻没有想到，因此我想在这篇文章中谈谈 python 中这样一个简单的概念。

在 python 中，我们可以从各种来源导入数据，如 csv 文件、excel 文件、txt 文件、SAS、各种数据库。所以，我这里的想法是总结用于在 python 中导入任何类型文件的各种代码。

为了执行这个，**我们需要安装一个名为“pandas”**的包。通常，当我们在系统中安装了 anaconda 时，不需要单独安装 pandas，因为它是与 anaconda 一起内置的。唯一需要的是，在运行本文中要遵循的任何其他 python 代码之前，使用下面的命令导入 pandas 库。

**进口熊猫当 pd**

# *1>****CSV-逗号分隔值文件。***

**Read_csv()函数**可以用来读取 csv 文件。

进口熊猫作为 pd

data = **pd.read_csv** (“文件路径/文件名. csv”)

**注意:**当我们从系统目录中指定文件路径时，它带有一个反斜杠(\)。但是 python 不能识别它，为了避免同样的情况，我们需要将单反冲改为正斜杠(/)或双反斜杠(\\)

例如:

data = PD . read _ CSV(' C:/Users/安珠/桌面/文档/文件 1.csv ')

# ***2>CSV-逗号分隔值文件，不带标题。***

由于我们从各种来源收集数据，大多数时候我们没有得到所需的格式，文件中的数据可能有垃圾值、丢失值，有时数据甚至可能没有标题(列名)等。

虽然我们总是可以在 csv 文件本身中更正数据，例如:我们可以在 csv 文件本身中添加列名，以防 header 丢失，但在 python 中也有一种方法可以做到这一点。

进口熊猫作为 pd

data = **pd.read_csv** (“文件路径/文件名. csv”， **header = None，names =** ['列 1 '，'列 2 '，'列 3 '，'列 4 '，'列 5'])

在上面的代码中，“header= None”告诉 python 数据缺少标题(列名)。

' names = ['col names']是一个用于向数据添加列名的函数。

例如:

data = PD . read _ CSV(' C:/Users/安珠/Desktop/Documents/file1.csv '，header = None，names = ['ID '，' Fname '，' Lname '，' Subject '，' Marks'])

我们也可以使用下面的代码做同样的事情:

Data.columns = ['ID '，' Fname '，' Lname '，' Subject '，' Marks']

# 3 >**CSV-逗号分隔值文件。**

假设我只需要导入指定数量的行和列或者跳过一些行，我们可以使用下面的代码。

进口熊猫作为 pd

data = **pd.read_csv** (“文件路径/文件名. csv”， **nrows=10，usecols = (2，5，7，8))**

在上面的代码中，nrows = 10 将只从 csv 文件中导入前十行，并为 usecols 函数中提到的变量/列检索信息。

这里，usecols = (2，5，7，8)将只获取列 2，5，7，8 的行。

例如:

data = PD . read _ CSV(' C:/Users/安珠/桌面/文档/文件 1.csv '，nrows=10，usecols = (2，5，7，8))

**跳行:**

进口熊猫作为 pd

data = **pd.read_csv** (“文件路径/文件名. csv”， **skiprows=10** )

函数 skiprows = 10 将跳过数据的前十行。

# 4 **> Excel 文件。**

**Read_excel()函数**可以用来读取 excel 文件。

进口熊猫作为 pd

data = **pd.read_excel** (“文件路径/文件名. xlsx”)

例如:

data = PD . read _ excel(' C:/Users/安珠/桌面/文档/文件 1.xlsx ')

# 5 **> Excel 文件—有时一个 Excel 文件可以有多个工作表。**

如果我们需要处理 excel 文件中的任何特定工作表，那么我们可以指定工作表的名称。假设，我需要从 excel 文件中导入第二张工作表。

进口熊猫作为 pd

data =**PD . read _ excel**(" file path/filename . xlsx "， **sheetname = 'sheet2')**

例如:

data = PD . read _ excel(' C:/Users/安珠/桌面/文档/文件 1.xlsx '，工作表名称= '工作表 2 ')

函数“sheetname = 'sheet2 '”将在 python 中导入 sheet2。

注意:如果没有提到工作表名称，默认情况下，第一个工作表将从包含多个工作表的 excel 文件中导入。

# 6 >任何文件(CSV/EXCEL/TXT 等)

假设文件中的数据有一些特殊字符或者像点(.)，下划线(_)，问号(？)等和**我们想在导入文件**时将它们指定为缺失值，下面的代码可以帮助我们做到这一点。

进口熊猫作为 pd

data =**PD . read _ excel**(" file path/filename . xlsx "， **na_values** = [' . ", '_' , '?'])

例如:

data = PD . read _ excel(' C:/Users/安珠/桌面/文档/文件 1.xlsx '， **na_values = [' . ', '_' , '?'])**

' **na_values** '选项中的所有字符将被视为缺失值。

# 7 >文本文件:

**Read_table()函数**可以用来读取文本文件。

进口熊猫作为 pd

data = **pd.read_table** (“文件路径/文件名. txt”)

例如:

data = PD . read _ table(' C:/Users/安珠/桌面/文档/文件 1.txt ')

# **8 >文本文件由制表符分隔/分隔。**

进口熊猫作为 pd

data = **pd.read_table** (“文件路径/文件名. txt”， **sep = "\t"** )

例如:

data = PD . read _ table(' C:/Users/安珠/桌面/文档/文件 1.txt '，sep = '\t ')

# 9 >由空格分隔的文本文件。

进口熊猫作为 pd

data =**PD . read _ table**(" file path/filename . txt "， **sep = "\s+"** )

例如:

data = PD . read _ table(' C:/Users/安珠/桌面/文档/文件 1.txt '，sep = '\s+')

**如果我们只想导入几列的数据，那么我们可以使用‘names’函数:**

data = PD . read _ table(' C:/Users/安珠/Desktop/Documents/file1.txt '，sep = '\s+'，names = ['ID '，' Fname '，' Lname '，' Subject '，' Marks'])

# 10 >包含二进制数据的文本文件。

进口熊猫作为 pd

data =**PD . read _ table**(" file path/**filename . dat**")

data =**PD . read _ table**(" file path/**filename . dat "**，sep = "\t" ) —数据由 tab 分隔。

data =**PD . read _ table**(" file path/**filename . dat**"，sep = " \ s+")-数据由空格分隔。

# 11 >来自 URL。

为了从任何 url 导入 python 中的文件，只需包括 URL 链接，可以根据存储数据的文件类型使用 read_xxxx 命令。

进口熊猫作为 pd

data =**PD . read _ CSV**(“https://www . objective quiz . com//doc . CSV”)

data =**PD . read _ exce**l(" https://www . objective quiz . com//doc . xlsx ")

data =**PD . read _ table**(“https://www . objective quiz . com//doc . txt”)

data =**PD . read _ table**(“https://www . objective quiz . com//doc . dat”)

# 12> SAS 文件。

要在 python 中导入 SAS 文件，可以使用' read_sas '函数。

进口熊猫作为 pd

data =**PD . read _ SAS**(" dataset . library ")

例如:

data =**PD . read _ SAS**(" cars . SAS user ")

# 13 >数据库。

假设我们想从任何数据库(如 SQL server)中导入存储在表中的数据，我们需要首先建立到 SQL server 的连接，这需要服务器名称、用户 id、密码和其他数据库信息。

**pd.read_sql_query 同样可以使用。**

**这里，我们还需要导入另一个库‘py odbc’来建立与服务器的连接。**

进口熊猫作为 pd

导入 pyodbc

conc = py odbc . connect(" Driver = { SQL Server }；Server =服务器名称；UID =用户的 ID，PWD =用户的密码；Database =数据库名称；")

data =**PD . read _ SQL _ query**(' select * from database . tablename '，conc)