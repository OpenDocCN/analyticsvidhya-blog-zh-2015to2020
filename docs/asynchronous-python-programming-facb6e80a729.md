# 异步 Python 编程

> 原文：<https://medium.com/analytics-vidhya/asynchronous-python-programming-facb6e80a729?source=collection_archive---------17----------------------->

今天，我将向你们展示一些不同的东西。我仍然试图在几乎每一个合适的主题上使用这个主题，并因此掌握它的是使用 Python 的异步编程。

在我的公司里，我们管理着 195 个读写 Oracle 数据库(以及它们的本地和远程备用数据库),在某些情况下，我们必须查询每一个数据库。当然，我们已经安排了任务来整合监控数据库中的数据。然而，我们的库存不足以满足我们和/或我们客户的需求。

我们可以从所有数据库中连续收集数据，也可以同时从所有数据库中收集数据。

为了比较公平，我将使用 Python 来测量串行处理时间。

让我们看一看，

```
import time
import cx_Oracle
import concurrent.futurest1 = time.perf_counter()query= f"""select distinct db_name, tns_host, tns_port, tns_service from inv_schema.inv_table"""]mapdsn = cx_Oracle.makedsn('inv_db_dns', 'inv_db_port',service_name='inv_db_service')
con1 = cx_Oracle.connect("inv_db_user", "inv_db_passwd", mapdsn, encoding="UTF-8")
cursor1 = con1.cursor()
cursor1.execute(query)strmappeddsn = [(element[0], f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={element[1]})(PORT={element[2]}))(CONNECT_DATA=(SERVICE_NAME={element[3]})))") for element in cursor1.fetchall()]def connect_and_execute(dsn): 
    querylist = ["""select count(*) from dba_role_privs"""]
    try:
        con = cx_Oracle.connect("inv_db_user", "inv_db_passwd", dsn[1], encoding="UTF-8")
        if not con: return
        else:
            cursor = con.cursor()
            x = []
            for query in querylist:
                cursor.execute(query)
                try:
                    x.append(cursor.fetchall())
                except Exception as e:
                    pass
                con.close()
            print (str(dsn[0]), str(x))

    except Exception as e:
        print(str(dsn[0]) + "\t" + str(e))
        passt2 = time.perf_counter()

for x in strmappeddsn:
    connect_and_execute(x)

t3 = time.perf_counter()

print(t2-t1)
print(t3-t2)
```

如下所示，对于 195 个数据库，仅串行处理时间(t3 — t2)就花费了大约 100.5 秒。

![](img/0b767ac9f6258e695a0927b6a6cb66bb.png)

另一方面，

```
import time
import cx_Oracle
import concurrent.futurest1 = time.perf_counter()query= f"""select distinct db_name, tns_host, tns_port, tns_service from inv_schema.inv_table"""]mapdsn = cx_Oracle.makedsn('inv_db_dns', 'inv_db_port',service_name='inv_db_service')
con1 = cx_Oracle.connect("inv_db_user", "inv_db_passwd", mapdsn, encoding="UTF-8")
cursor1 = con1.cursor()
cursor1.execute(query)strmappeddsn = [(element[0], f"(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={element[1]})(PORT={element[2]}))(CONNECT_DATA=(SERVICE_NAME={element[3]})))") for element in cursor1.fetchall()]def connect_and_execute(dsn): 
    querylist = ["""select count(*) from dba_role_privs"""]
    try:
        con = cx_Oracle.connect("inv_db_user", "inv_db_passwd", dsn[1], encoding="UTF-8")
        if not con: return
        else:
            cursor = con.cursor()
            x = []
            for query in querylist:
                cursor.execute(query)
                try:
                    x.append(cursor.fetchall())
                except Exception as e:
                    pass
                con.close()
            **return (str(dsn[0]), str(x))**

    except Exception as e:
        print(str(dsn[0]) + "\t" + str(e))
        passt2 = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(connect_and_execute, strmappeddsn)
        for result in results: 
            output.append(result)t3 = time.perf_counter()[print(_) for _ in output]
print(t2-t1)
print(t3-t2)
```

在这两个 Python 脚本中，我都使用了相同的函数来连接 Oracle 数据库和执行 SQL，唯一不同的是，在异步版本中，我返回了 SQL 的输出，而不是 print。

对于异步编程，我使用并发库。根据[https://docs.python.org/3/library/concurrent.futures.html](https://docs.python.org/3/library/concurrent.futures.html)，在创建一个执行器后我们可以使用**映射**函数来调用异步调用。此映射函数与内置映射函数的不同之处在于:

*   这些*可重复项*被立即收集，而不是被懒洋洋地收集。
*   *func* 是异步执行的，可以同时调用几次 *func* 。

此外，在 Python 3 . 7 . 1 版的官方 Python 文档中，默认的最大工作线程数(最大工作线程数的计算从 Python 3.5 版到 3.8 版一直没有改变)是“**机器上的处理器数乘以 5** ”。所以，

```
>>> import os
>>> print(os.cpu_count())
8
>>>
```

我的最大工人数是 40。

好吧，但是，结果呢？我不会再引起好奇心了。对于 195 个数据库，时间大约为 3.68 秒。

![](img/813feda4d13edad3ea01ae3a873f948d.png)

100,4983596 / 3,6819677 = 27,2947398750945

大约好 26 倍:)。很酷，是吧？

在接下来的故事中，我将尝试优化最大工作人员数量，并突破 Oracle 数据库云服务器的极限。

感谢阅读！

乌穆特·泰京