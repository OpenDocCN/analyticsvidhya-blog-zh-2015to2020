# 第 3.6 部分！！使用 Python 将数据帧转换为 PostgreSQL

> 原文：<https://medium.com/analytics-vidhya/part-3-6-pandas-dataframe-to-postgresql-using-python-ec80cb33ca4a?source=collection_archive---------3----------------------->

## 使用 copy_from()和 StringIO 将批量数据插入 PostgreSQL 数据库

![](img/cc927d95daa51df385e692e98ede6439.png)

# 先决条件

**Python 3.8.3 :** [蟒蛇下载链接](https://www.anaconda.com/products/individual)

**PostgreSQL 13:**T6**下载链接**

**Psycopg2 :** 安装 **Psycopg2** 使用命令: **pip 安装 psycopg2**

# 目标

本文的主要目标是逐步学习使用 StringIO 方法的 ***copy_from()的工作代码。***

# 第 1 步:准备或识别您的数据

首先，准备或确定要导入 PostgreSQL 数据库的 CSV 文件。例如，我们从 GitHub 加载虹膜数据。

```
import os
***# import sys to get more detailed Python exception info***
import sys
***# import the connect library for psycopg2***
import psycopg2
***# import the error handling libraries for psycopg2***
from psycopg2 import OperationalError, errorcodes, errors
import psycopg2.extras as extras
import pandas as pd
from io import StringIO
import numpy as npirisData = pd.read_csv('[https://raw.githubusercontent.com/Muhd-Shahid/Learn-Python-Data-Access/main/iris.csv',index_col=False](https://raw.githubusercontent.com/Muhd-Shahid/Learn-Python-Data-Access/main/iris.csv',index_col=False))
irisData.head()
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
    return conn**# Define function using copy_from() with StringIO to insert the 
# dataframe**
def copy_from_dataFile_StringIO(conn, datafrm, table):

  # save dataframe to an in memory buffer
    buffer = StringIO()
    datafrm.to_csv(buffer, header=False, index = False)
    buffer.seek(0)

    cursor = conn.cursor()
    try:
        cursor.copy_from(buffer, table, sep=",")
        conn.commit()
        print("Data inserted using copy_from_datafile_StringIO() successfully....")
    except (Exception, psycopg2.DatabaseError) as error:
        # pass exception to function
        show_psycopg2_exception(err)
        cursor.close()
    cursor.close()
```

# 步骤 4:执行主要任务

```
***# Connect to the database***
conn = connect(conn_params_dic)
conn.autocommit = True
***# Run the copy_from_dataFile_StringIO() method***
copy_from_dataFile_StringIO(conn, irisData, 'irisCopyFromFileStrIO')
```

# 第五步:查询数据库，检查我们的工作

让我们查询数据库，以确保我们插入的数据已被正确保存。

```
***# Prepare sql query***
sql = "SELECT * FROM irisCopyFromFileStrIO" 
cursor = conn.cursor()
***# Execute cursor***
cursor.execute(sql)
***# Fetch all the records***
tuples = cursor.fetchall()
***# list of columns*** cols = list(irisData.columns)
irisdf = pd.DataFrame(tuples,columns=cols) 
***# Print few records***
print(irisdf.head())
***# Close the cursor***
cursor.close()
***# Close the connection***
conn.close()
```

![](img/3cfbed662a8951cb249ceaecd3d93898.png)

**结论:**我们的**部分 3.6** 到此结束。在本教程中，我们学习了如何使用 copy_from()和 StringIO 方法将批量数据插入 PostgreSQL 数据库。

本文的所有代码都可以作为 GitHub *上的 [***Jupyter 笔记本获得。***](https://github.com/Muhd-Shahid/Learn-Python-Data-Access/tree/main/PostgreSQL)*

> **接下来** [***第 3.7 部分***](https://shahid-dhn.medium.com/part-3-7-pandas-dataframe-to-postgresql-using-python-6590fda63f41) **:使用 *to_sql(alchemy)()* 方法将批量数据插入使用 Python 的 PostgreSQL 数据库**
> 
> ***之前的学习:***
> 
> [***第一部分***](https://shahid-dhn.medium.com/pandas-dataframe-to-postgresql-using-python-part-1-93f928f6fac7) ***:简介、连接&数据库创建***
> 
> [**第二部分**](https://shahid-dhn.medium.com/pandas-dataframe-to-postgresql-using-python-part-2-3ddb41f473bd) **使用 Python 在 PostgreSQL 数据库中创建表**
> 
> [***第 3.1 部分***](https://shahid-dhn.medium.com/part-3-1-pandas-dataframe-to-postgresql-using-python-8a3e3da87ff1) ***:* 使用 executemany()将批量数据插入 PostgreSQL 数据库**
> 
> [***第 3.2 部分***](/analytics-vidhya/part-3-2-pandas-dataframe-to-postgresql-using-python-8dc0b0741226) ***:* 使用 execute_batch()将批量数据插入 PostgreSQL 数据库**
> 
> [***第 3.3 部分***](https://shahid-dhn.medium.com/part-3-3-pandas-dataframe-to-postgresql-using-python-57e68fe39385) ***:* 使用 Python 将使用 execute_values()方法的批量数据插入 PostgreSQL 数据库**
> 
> [***第 3.4 部分***](https://shahid-dhn.medium.com/part-3-4-pandas-dataframe-to-postgresql-using-python-d94e644a332) ***:* 使用 mogrify()将批量数据插入 PostgreSQL 数据库**
> 
> [***第 3.5 部分***](https://shahid-dhn.medium.com/part-3-5-pandas-dataframe-to-postgresql-using-python-d3bc41fcf39) ***:* 使用 copy_from()将批量数据插入 PostgreSQL 数据库**

保持积极的态度！！注意安全！！继续学习:))

**感谢您的阅读！！**