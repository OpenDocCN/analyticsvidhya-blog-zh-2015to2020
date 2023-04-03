# 第 5.2 部分！！使用 Python 将数据帧转换为 PostgreSQL

> 原文：<https://medium.com/analytics-vidhya/part-5-2-pandas-dataframe-to-postgresql-using-python-450607b763b4?source=collection_archive---------17----------------------->

## 用 Python 提取、转换和加载 PostgreSQL 数据

![](img/0b9f9246595e94fbfe82475c76257ced.png)

# 先决条件

**Python 3.8.3 :** [蟒蛇下载链接](https://www.anaconda.com/products/individual)

**PostgreSQL 13:**T6**下载链接**

**Psycopg2 :** 安装 **Psycopg2** 使用命令: **pip install psycopg2，petl**

# 目标

通常，我们通过将 csv 文件导入 pandas 数据帧来创建训练和测试数据，但当我们在数据库服务器中存储大量数据时，我们需要一种方法来直接从数据库服务器将其提取到 pandas 数据帧中。为了这个目标，我们将一步一步地学习不同方法的工作代码。

# 步骤 1:导入库

```
# import sys to get more detailed Python exception info
import sys
# import the connect library for psycopg2
import psycopg2
# import the error handling libraries for psycopg2
from psycopg2 import OperationalError, errorcodes, errors
import psycopg2.extras as extras
from sqlalchemy import create_engine
import pandas as pd# Extract Trabsforl & Load
import petl as etl
```

# 步骤 2:指定连接参数

```
**# Note: please change your database, username & password as per your own values** conn_params_dic = {
    "host"      : "localhost",
    "database"  : "irisdb",
    "user"      : "postgres",
    "password"  : "Passw0rd"
}
```

# 第三步:支持功能

```
***# Define a function that handles and parses psycopg2 exceptions*
def show_psycopg2_exception(err):**
    **# get details about the exception**
    err_type, err_obj, traceback = sys.exc_info()   ** 
    # get the line number when exception occured**
    line_n = traceback.tb_lineno    
    **# print the connect() error**
    print ("\npsycopg2 ERROR:", err, "on line number:", line_n)
    print ("psycopg2 traceback:", traceback, "-- type:", err_type) 
    **# psycopg2 extensions.Diagnostics object attribute**
    print ("\nextensions.Diagnostics:", err.diag)   ** 
    # print the pgcode and pgerror exceptions**
    print ("pgerror:", err.pgerror)
    print ("pgcode:", err.pgcode, "\n")***# Define a connect function for PostgreSQL database server*
def connect(conn_params_dic):**
    conn = None
    try:
        print('Connecting to the PostgreSQL...........')
        conn = psycopg2.connect(**conn_params_dic)
        print("Connection successfully..................")

    except OperationalError as err:
        **# passing exception to function**
        show_psycopg2_exception(err)       ** 
        # set the connection to 'None' in case of error**
        conn = None
    return conn
```

# 步骤 4:将 PostgreSQL 数据加载到 CSV 文件中:提取、转换和加载 PostgreSQL 数据

我们可以使用 petl 来提取、转换和加载 PostgreSQL 数据。在本例中，我们提取 PostgreSQL 数据，按物种列对数据进行排序，并将数据加载到一个 CSV 文件中。

```
# Connecting to PostgreSQL Data
conn = connect(conn_params_dic)# Create a SQL Statement to Query PostgreSQL
#sql = "SELECT * FROM iris WHERE species = 'testing'"
sql = "SELECT * FROM iris "extractData = etl.fromdb(conn,sql)

extractData.head()
```

# 改变

在这里，您可以在加载后对数据执行不同的运算。例如，您可以执行以下操作

1.  连接表，
2.  排序，
3.  填充缺失值
4.  改造桌子等。

```
transformData = etl.sort(extractData,'species')
```

# 加载(写入数据)

在所有的数据操作之后，现在可以将你的数据保存在你的磁盘中，以备报告或以后使用。

```
etl.tocsv(transformData,'../Learn Python Data Access/iris_v1.csv')
```

**结论:**我们的 5.2 部分到此结束。在本教程中，我们学习了如何提取、转换和加载数据。

这篇文章的所有代码都可以作为 GitHub *上的 [***Jupyter 笔记本获得。***](https://github.com/Muhd-Shahid/Learn-Python-Data-Access/tree/main/PostgreSQL)*

> ***以前的学习:***
> 
> [**第一部分**](https://shahid-dhn.medium.com/pandas-dataframe-to-postgresql-using-python-part-1-93f928f6fac7) **:简介、连接&数据库创建**
> 
> [**第二部分**](https://shahid-dhn.medium.com/pandas-dataframe-to-postgresql-using-python-part-2-3ddb41f473bd) **使用 Python 在 PostgreSQL 数据库中创建表**
> 
> [**第 3.1 部分**](https://shahid-dhn.medium.com/part-3-1-pandas-dataframe-to-postgresql-using-python-8a3e3da87ff1) **:使用 executemany()将批量数据插入 PostgreSQL 数据库**
> 
> [**第 3.2 部分**](/analytics-vidhya/part-3-2-pandas-dataframe-to-postgresql-using-python-8dc0b0741226) **:使用 execute_batch()将批量数据插入 PostgreSQL 数据库**
> 
> [**第 3.3 部分**](https://shahid-dhn.medium.com/part-3-3-pandas-dataframe-to-postgresql-using-python-57e68fe39385) **:使用 Python** 将使用 execute_values()方法的批量数据插入 PostgreSQL 数据库
> 
> [**第 3.4 部分**](https://shahid-dhn.medium.com/part-3-4-pandas-dataframe-to-postgresql-using-python-d94e644a332) **:使用 mogrify()将批量数据插入 PostgreSQL 数据库**
> 
> [**第 3.5 部分**](https://shahid-dhn.medium.com/part-3-5-pandas-dataframe-to-postgresql-using-python-d3bc41fcf39) **:使用 copy_from()将批量数据插入 PostgreSQL 数据库**
> 
> [**第 3.6 部分**](https://shahid-dhn.medium.com/part-3-6-pandas-dataframe-to-postgresql-using-python-ec80cb33ca4a) : **使用 copy_from()和 StringIO 将批量数据插入 PostgreSQL 数据库**
> 
> [**第 3.7 部分**](/analytics-vidhya/part-3-7-pandas-dataframe-to-postgresql-using-python-6590fda63f41) : **使用 to_sql()(alchemy)将批量数据插入 PostgreSQL 数据库**
> 
> [**第四部分**](https://shahid-dhn.medium.com/part-4-pandas-dataframe-to-postgresql-using-python-8ffdb0323c09) : **使用 Python 将批量 CSV 数据导入 PostgreSQL 的方法比较**
> 
> [**第 5.1 部分**](https://shahid-dhn.medium.com/part-5-1-pandas-dataframe-to-postgresql-using-python-e2588e65c235) : **如何从 PostgreSQL 读取数据到 Pandas DataFrame？**

***参考:petl:***[***petl—提取、转换和加载(数据表)— petl 1.6.8 文档***](https://petl.readthedocs.io/en/stable/)

保持积极的态度！！注意安全！！继续学习:))

**感谢您的阅读！！**