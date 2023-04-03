# 深度存储过程

> 原文：<https://medium.com/analytics-vidhya/stored-procedure-in-depth-by-sagar-jaybhay-2020-sagar-jaybhay-e8319ca81a81?source=collection_archive---------14----------------------->

# 什么是存储过程？

存储过程是一组 SQL 语句。我们将每次都要使用的 SQL 语句分组，并为其指定一个名称。这样做的结果是，你准备了一个 SQL 代码并保存它。当你再次需要这个的时候，你只需要叫出你给的名字。

假设您有一个查询**select * from employee；**这个查询你要反复保存它为一个过程。

不能在 select 子句和 where 子句中使用存储过程。

## 过程的一般语法:

```
CREATE PROCEDURE procedure_name
AS
begin
sql_statement
end
GO;
```

**举例**

```
create procedure GetAllEmployee    --Create stored procedure syntax
as 
begin
select * from Employee;
end
```

1.  在上面的查询中，GetAllEmployee 是我们给定的[过程](https://docs.microsoft.com/en-us/sql/relational-databases/stored-procedures/create-a-stored-procedure)的名称。请避免在过程名称中使用 sp 前缀，这是 Microsoft 建议的，因为 Microsoft 给出的内置过程是以字母 sp 开头的。
2.  As 关键字用于分隔存储过程的标题和主体。
3.  之后，我们有 begin 和 end 语句，如果只有一个 SQL 语句，那么 begin 和 end 关键字是可选的。但是，即使只有一条 SQL 语句，也应该有开头和结尾。
4.  create Procedure 是创建过程的语法。

# 如何执行存储过程？

调用过程有不同的方法，我们有下面的方法。

1.  存储过程名称，然后按 F5 在 SQL server management studio 上运行
2.  Exec 存储过程名称
3.  执行存储过程名称

# 如何删除存储过程？

您可以在存储过程名称中使用 drop procedure 关键字。

```
drop procedure GetAllEmployee; -- drop the store procedure
```

# 如何在 SQL server 中获取存储过程文本？

使用 **sp_helptext** 内置过程并传递过程名，然后您将获得所有存储过程语法。

```
sp_helptext GetAllEmployee  -- get the text of stored procedure
```

# 如何改变 SQL server 中的存储过程？

您可以将 alter 关键字与过程名一起使用，并更改您想要更改的 SQL 语句，然后运行该语句，您的过程将被更改。

```
alter procedure GetAllEmployee  -- alter a stored procedure
as 
begin
select top 10 * from Employee;
end
```

# 如何在 SQL server 中创建参数化存储过程？

要创建参数化过程，请使用以下通用语法。

```
Create procedure Proc_name @param1 data_type,@para2 data_type
As
Begin
SQL statement
End;
```

**例子**

```
create proc GetEmpBasedOnParameter -- create parametrize stored procedure
@departmentid int,
@gender varchar(10)
as
begin
select * from Employee where DepartmentID=@departmentid and Gender=@gender;
end
```

1.  在上面的代码语法中，我们有@departmentid 是整数参数，而@geneder 是第二个具有 varchar 数据类型的参数。
2.  其余的过程语法是相同的，但是如果您看到一个 select 语句，我们在 where 子句中使用@departmentid 和@geneder 参数，并且当我们调用这个过程时传递这个参数。
3.  如果不为参数化过程传递参数，将会出现错误。
4.  我们可以改变参数的顺序，但是我们需要在过程调用中显式地定义它们。

```
exec GetEmpBasedOnParameter @gender='female',@departmentid=10      --run stored procedure syntax with parameters position not required
```

# 带输出参数的存储过程

```
create procedure GetGenderwisecount  -- procedure with an output parameter
@gender varchar(10),
@empCount int out
as
begin
select @empCount = count(empid) from Employee where Gender=@gender
end;
```

在上面的过程中，我们创建了一个带有输出参数的参数化过程。这个过程将性别作为输入，并产生该性别的总计数作为输出，因此声明为 **@empcount** int out 这是我们的输出参数，我们可以声明为 **@empcount** int output，就像两者都将产生相同的结果。

# 如何调用返回输出参数的存储过程？

如果我们的存储过程返回输出参数，那么我们需要在一个变量中捕获输出。所以当我们调用这个过程时，我们需要声明一个标量变量，并获取输出参数。

在使用声明的语法声明输出参数之后，在此之后，当您调用 execute 语句时，您需要指定不带 or output 的变量，否则它将不会在我们声明的输出参数中获得结果。如果不声明不带参数的变量，值将为 null。

声明输出参数的一般语法是

```
@product_count INT OUTPUT

declare @empcount int
exec GetGenderwisecount 'feMale',@empcount out
print(@empcount)
```

> *我们可以在存储过程中有多个输出参数。*

# 存储过程:输出参数与返回值

# 返回值

创建一个将返回值的存储过程。

```
create procedure GetGenderwisecount2  -- procedure with a return value
@gender varchar(10)
as
begin
return (select count(empid) from Employee where Gender=@gender)
end;
```

在上面的过程中，我们使用了 return 语句，在括号中，我们编写了一个基于性别计算 emp 的查询，在这个过程中，我们使用了这些括号，因为它将首先执行，然后将结果传递给 return 语句。

为了运行上面的过程，我们使用下面的代码

```
declare @count int
exec @count= GetGenderwisecount2 'male'
print(@count)
```

在这里，我们在@count 变量中收集返回值，并且在 create procedure 语法中没有使用 output 参数。上面的调用过程工作得非常好，因为它返回数字值。

永远记住，当你运行你的存储过程时，它将返回一个整数值，这是过程的状态。

现在我们尝试使用 return 语句返回字符串值。

```
Create procedure GetEmployeeNamebyID
@id int 
as
begin
return (select Employee.full_name from Employee where EmpID=@id);
end;
```

在上面的 create stored procedure 语法中，我们将 if 作为输入参数传递，并尝试返回 name 值，但是当您尝试执行此过程时，它会抛出一个错误

```
exec GetEmployeeNamebyID 1
```

消息 245，级别 16，状态 1，过程 GetEmployeeNamebyID，第 5 行[批处理开始行 91]

将 varchar 值“Harcourt Loalday”转换为数据类型 int 时，转换失败

> *所以要记住存储过程的返回值始终是一个整数。*

# 存储过程的优点

1.  存储过程的最大好处是它们能够重用执行计划。执行计划意味着当您启动一个查询时，SQL server 首先检查查询的语法，然后编译该查询，最后生成一个执行计划。简而言之，执行计划是为了从数据库中获取数据，这是检索数据的最佳方式。您使用的是存储过程，因此 SQL server 会缓存该执行计划并将其存储在内存中，当您再次调用该存储过程时，它不会执行语法检查、编译查询和生成执行计划等所有过程，而是会使用缓存的执行计划，因此最终会提高性能。相比之下，如果使用来查询，它也会生成缓存计划查询，但查询中的小变化会导致所有步骤重新执行。
2.  它驻留在服务器上，因此任何想要使用该过程的应用程序都可以使用它。从而实现可维护性和可重用性。
3.  它提供了良好的安全性。
4.  它避开了 SQL 注入的攻击。

*原载于 2020 年 1 月 7 日*[*【https://sagarjaybhay.com】*](https://sagarjaybhay.com/stored-procedure-in-depth-by-sagar-jaybhay-2020/)*。*