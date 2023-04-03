# SQL Server 数据库设置和通过 Python 连接

> 原文：<https://medium.com/analytics-vidhya/sql-server-database-setup-and-connecting-through-python-f0cac42fecc8?source=collection_archive---------19----------------------->

SQL Server 2014 的环境设置:

首先，下载微软的 [**。NET Framework 3.5**](https://www.microsoft.com/en-us/download/details.aspx?id=21) 并安装。安装后，您可能需要重新启动系统。

![](img/7376e8d4980a0051616edd2b1eaa8d5e.png)![](img/0361e18410c543e8ba1d4faa85acecdf.png)

*   然后下载[微软 SQL Server 2014 Express](https://www.microsoft.com/en-us/download/details.aspx?id=42299)
*   检查您的系统是否满足所有要求，或者安装缺少的要求/软件。

![](img/a68d63103f8d0b4b98392c04b08828af.png)

*   对于这个项目，我使用的是 64 位机器。因此，我选择了 64 位软件/文件进行安装。
*   下载文件后。解压文件(选择解压文件的目录)。

![](img/03f4bbc7d3a3e793de2d20fcc63568e0.png)

*   一旦被提取出来。安装过程直接开始。然后选择规划选项，然后选择系统配置检查器并运行它，以检查配置是否满足所有要求。如果系统满足所有要求，新窗口会自动打开和关闭。

![](img/6af1f60d3e17c9a8e2d3914bf62c657b.png)![](img/ff94df1d0e52bff6dccb850276dc1cd0.png)

*   然后继续安装
*   在“安装选项”下，选择“新建 SQL Server 独立安装”或“向现有安装添加功能”。

![](img/0cbc720676ea5a64032366a0d3cd5a69.png)

*   选择许可条款。然后点击下一步。

![](img/381cb77c43730ade56159662c5adf8b6.png)

*   可选服务选择选项使用 Microsoft Update 检查更新(推荐)

![](img/f31cdd129a64c701fb1e4be905bb4dfc.png)

*   选择您需要的所有功能。根据您的要求对目录进行更改。对于这一个，我使用默认的目录。点击下一步。

![](img/f28c930ff978bde1e65de386aefc2f54.png)

*   选择“命名实例”选项(默认名称)。对于这个安装，我使用名称 SQLExpress。你想取什么名字都可以。但是我们很注意间距。我不建议给实例留出空间。记下您给出的实例名称，因为您稍后将使用它通过 SSMS 连接到实例。点击下一步。

![](img/ea08a54104f40d50b62bb2e1d991db8b.png)

*   单击“下一步”进行服务器配置。里面没什么需要改的。
*   在“服务器配置”选项卡下的“数据库引擎配置”下，选择身份验证模式。为此，选择默认的 Windows 身份验证模式。

![](img/3831bc54ca4a53befbe28073e6e3544a.png)

*   然后转到 **FILESTREAM** 选项卡，选择所有选项，然后单击下一步。

![](img/04028842039675a4612cfcabde80e6be.png)

*   安装开始，安装完成后。检查所有功能是否安装正确。然后关上窗户。

![](img/48f6bd5e6878288056850160ab1ebbd4.png)

*   然后从 windows 开始菜单启动 SSMS，并以管理权限运行。

![](img/f44e4185e89a57314b039abf7191cfd6.png)

*   选择服务器类型为数据库引擎，您在安装 SQL server 时提供的服务器名称(服务器名称(或计算机名称)\实例名称)，身份验证。然后点击连接。

![](img/7d0594fad1fda4ab57e9cddafb6a32c9.png)

*   在连接到实例之后，展开数据库连接，您可以看到一些默认的系统数据库，如下所示。我们将为我们的项目使用主数据库。

![](img/74b688f041a079c0d47d2358c4eb0f9a.png)

*   进一步展开并运行查询，您可以找到如下所示的表值。

![](img/f825e81109d21eb5226520e658c91527.png)

*   我们的项目是连接到 SQL server 数据库，从特定的表中检索数据/表，并将其存储在. csv 格式的文件中，我们正在连接到本地 SQL Server 数据库。
*   我希望你们都已经安装了 [**蟒蛇分布**](https://www.anaconda.com/download/) 。如果没有安装，我请求您下载并安装它。我将在 Anaconda 发行版中使用 Jupyter 笔记本。如果你愿意，你也可以只安装 [**Jupyter 笔记本**](http://jupyter.org/install) 。
*   启动 Jupyter 笔记本，打开一个新的 python Jupyter 笔记本，如下图所示。

![](img/88ba2b275ca2e25112b00f9ed5674748.png)

*   下面是这些值的最终输出。

在 Python(通过 Jupyter Notebook)中，我所做的是连接到我的本地 SQL Server 数据库实例 SQLExpress，稍后运行 SQL 查询从系统数据库主服务器获取一个表，然后将该表存储到 SQL_data.csv 文件中。稍后使用 python 打开 SQL_data.csv 文件并显示表格。

![](img/0c230909452bf45ace818e6262ca0e7e.png)

Python(在 Jupyter 笔记本中)代码

*   安装库`pyodbc`以连接到服务器的 Python 代码(从 Jupyter 笔记本安装)。
*   将`pyodbc`和其他库导入 python。

`import pyodbc as connector`

`import pandas as pd`

`import csv`

*   下面是如何连接到 SQL Server 的 python 代码。

`driver = connector.connect("Driver={SQL Server Native Client 11.0};"`

`Server=<Server-name>\<Instance-name>;"`

`"Database=<Database-Name>;"`

`"Trusted_COnnection=yes;")`

*   编写 SQL 查询将数据从数据库导入 python

`table = pd.read_sql_query('<SQL Query>',driver)`

*   打印表格。

`print(table)`

*   将表格存储到. csv 格式的文件中。

`dfCSV = table.to_csv('<Filename>.csv',index=False)`

*   读取 python 中保存的`<Filename>.csv`文件。

`pd.read_csv('<Filename>'.csv)`

如需示例代码文件，请点击此处的。

![](img/7e97cde71015ca9cb2e7634ca84db0d4.png)

[Jupyter 笔记本](https://github.com/VVRChilukoori/SQL-server-to-Python/blob/master/code-files/Connecting%20to%20SQL%20Server%20database%20from%20python.ipynb)文件为脚本

[](https://github.com/VVRChilukoori/SQL-server-to-Python/blob/master/code-files/Connecting%20to%20SQL%20Server%20database%20from%20python.ipynb) [## VVRChilukoori/SQL-服务器到 Python

### permalink dissolve GitHub 是 4000 多万开发人员的家园，他们一起工作来托管和审查代码，管理…

github.com](https://github.com/VVRChilukoori/SQL-server-to-Python/blob/master/code-files/Connecting%20to%20SQL%20Server%20database%20from%20python.ipynb)