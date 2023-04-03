# 在 SQL Server 中提取光标

> 原文：<https://medium.com/analytics-vidhya/fetch-cursor-in-sql-server-2684d2c7d36?source=collection_archive---------4----------------------->

当需要逐行提交数据时，使用 SQL 游标。尽管性能不佳，我们仍然使用游标进行数据库开发和报告。

基本提取光标步骤:

1-声明游标。

2-你用光标做什么？编写将要进行逐行操作的 select 语句。

3-打开光标。

4-Write fetch next 语句。将光标中的特定值赋给变量。

5-While 循环。开始和继续数据处理的条件。

6-您应该在 while 循环中使用 fetch next 语句。

7-关闭游标，然后用 deallocate 语句销毁游标。

样品-1

```
use AdventureWorks2017
goDECLARE Location_Cursor CURSOR FOR
select LocationID,Name,CostRate 
from Production.Location
OPEN Location_Cursor;
FETCH NEXT FROM Location_Cursor;
WHILE @@FETCH_STATUS = 0
 BEGIN
 FETCH NEXT FROM Location_Cursor;
 END;
CLOSE Location_Cursor;
DEALLOCATE Location_Cursor;
GO
```

样品-2

```
use masterdeclare @database_name nvarchar(100);
declare @lgname  nvarchar(100);declare crs cursor for
select original_login_name, DB_NAME(database_id) as db_nm
from sys.dm_exec_sessions where is_user_process=1;open crsfetch next from crs into @lgname, @database_namewhile @@FETCH_STATUS=0beginprint 'Login:'+cast(@lgname as nvarchar(100))+' Database Name:'+@database_namefetch next from crs into @lgname, @database_nameendclose crs
deallocate crs;
```

我们通常使用光标样本的这些给定默认选项(样本-1、样本-2)。有时可能会很贵。但是，还有其他选择吗？是的，有。并且在这个站点中有很多方法的讲解:[https://docs . Microsoft . com/en-us/SQL/t-SQL/language-elements/declare-cursor-transact-SQL？redirectedfrom = MSDN&view = SQL-server-ver 15](https://docs.microsoft.com/en-us/sql/t-sql/language-elements/declare-cursor-transact-sql?redirectedfrom=MSDN&view=sql-server-ver15)

您可以使用这个脚本来比较几个提取光标选项。允许您取消对选项的注释并查看持续时间。

```
drop table if exists tempdb.dbo.#tmpCursor
create table #tmpCursor(
id int identity(1,1),
obid int 
)
goDECLARE [@StartTime](http://twitter.com/StartTime) datetime,[@EndTime](http://twitter.com/EndTime) datetime 
SELECT [@StartTime](http://twitter.com/StartTime)=GETDATE()DECLARE [@i](http://twitter.com/i) INT = 1;

DECLARE cur CURSOR
 --LOCAL
 --LOCAL STATIC
 --LOCAL FAST_FORWARD
 --LOCAL FORWARD_ONLY
FOR
 SELECT c.[object_id] 
 FROM sys.objects AS c
 CROSS JOIN (SELECT TOP 200 name FROM sys.objects) AS c2
 ORDER BY c.[object_id];

OPEN cur;
FETCH NEXT FROM cur INTO [@i](http://twitter.com/i);

WHILE (@@FETCH_STATUS = 0)
BEGIN
 insert into #tmpCursor
 values([@i](http://twitter.com/i))
 FETCH NEXT FROM cur into [@i](http://twitter.com/i)
END

CLOSE cur;
DEALLOCATE cur;SELECT [@EndTime](http://twitter.com/EndTime)=GETDATE() 
SELECT DATEDIFF(ms,[@StartTime](http://twitter.com/StartTime),[@EndTime](http://twitter.com/EndTime)) AS [Duration in microseconds]
```

如果你选择快进，你会发现它比其他的要快。但是，根据它的良好性能。我不建议取光标操作。因为 Sql server 更适合**集合操作**。光标逐行操作**因此可能会很慢。它们使用**更多的内存**，有时**会导致阻塞**。**