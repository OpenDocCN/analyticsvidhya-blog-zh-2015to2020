# AWS Lambda (Python) + MS SQL Server —简单的方法

> 原文：<https://medium.com/analytics-vidhya/aws-lambda-python-ms-sql-server-the-easy-way-e7667d371cc5?source=collection_archive---------0----------------------->

## 使用 Lambda 层从 AWS Lambda 连接到 MS SQL 的快速方法

![](img/3481d648b8c7e064c5d7251548f8ef4a.png)

[https://unsplash.com/photos/D7Q6JpFleK8](https://unsplash.com/photos/D7Q6JpFleK8)

# pyodbc

pyodbc 是一个开源 Python 模块，它使得访问 odbc 数据库变得简单。它实现了 [DB API 2.0](https://www.python.org/dev/peps/pep-0249) 规范，但是打包了更方便的 Pythonic。这是为了使用 python 桥接 odbc 连接而构建的。

## 第一步:

根据您的 python 版本，仅下载所需的文件。打开**的包装。所以**文件在工作目录下。

## 第二步:

现在打开你最喜欢的编辑器，开始编码。

```
import pyodbc def lambda_handler(event,context):
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=server_name;'
                          'Database=db_name;'
                          'Trusted_Connection=yes;')

    cursor = conn.cursor()
    cursor.execute('SELECT * FROM db_name.Table')

    for row in cursor:
        print(row)
```

## 第三步:

将上面的文件另存为**lambda _ function . py***，这样你的 lambda 就可以在你的代码中调用这个处理程序了。*

## *第四步:*

*压缩文件夹中的所有文件。这是给你的参考结构。*

```
*├─── <parent_folder>
├─────── lambda_function.py
├─────── libodbc.so.2
├─────── pyodbc.so*
```

## *第五步:*

*在 lambda 函数中上传 zip 文件。*

*完成了吗？不会。这只会为你加载 pyodbc 库，但不能连接到 MSSQL，因为这需要 OBBC 驱动程序。这意味着您必须遵循以下步骤:*

1.  *下载 Unix ODBC 驱动程序*
2.  *从源代码编译 Unix ODBC*
3.  *将二进制文件复制到应用程序目录*
4.  *下载 Microsoft SQL Server ODBC 驱动程序*
5.  *安装 Microsoft SQL Server ODBC 驱动程序*
6.  *为 unixODBC 复制 MS SQL 驱动程序二进制文件*
7.  *创造。驱动程序路径配置的 ini 文件*
8.  *将整个东西打包成一个. zip 文件*

*太多了？我已经包装好了—*

*[](https://github.com/kuharan/Lambda-Layers) [## 库哈兰/λ层

### python 的 AWS lambda 图层集合。通过在…上创建帐户，为 kuharan/Lambda 层开发做出贡献

github.com](https://github.com/kuharan/Lambda-Layers) 

如果需要，只需下载 zip 和示例代码。只需上传这个压缩文件作为 lambda 层，然后就大功告成了！

这是我在测试这一切时制作的一个快速视频。

您刚刚将 AWS Lambda 连接到 Microsoft SQL Server。现在你是无服务器计算的老板！

测试人员注意:

> 符合 python 3.7 和用于 SQL Server 的 ODBC 驱动程序 17。

相关的 **StackOverflow** 关于我最初做了什么以及现在处于什么位置的问题—

[](https://stackoverflow.com/questions/48016091/unable-to-use-pyodbc-with-aws-lambda-and-api-gateway/61555952#61555952) [## 无法将 pyodbc 与 AWS lambda 和 API 网关一起使用

### 我试图建立一个 AWS Lambda 功能使用 API 网关，利用 pyodbc python 包。我已经跟踪了…

stackoverflow.com](https://stackoverflow.com/questions/48016091/unable-to-use-pyodbc-with-aws-lambda-and-api-gateway/61555952#61555952)*