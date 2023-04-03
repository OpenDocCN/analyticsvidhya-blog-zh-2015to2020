# c 出口函数是什么？

> 原文：<https://medium.com/analytics-vidhya/what-is-the-c-exit-function-50bbb3121f6f?source=collection_archive---------20----------------------->

`exit`函数是 c 标准库的一部分，是在的`stdlib.h`头中定义的**。**

`stdlib.h`定义了一些类型、宏，它包含通用的实用函数，来执行数值转换、生成随机数、执行排序、分配内存或与环境交互。

`exit`功能用于与环境交互。它有以下签名。

```
void exit(int status);
```

`exit`功能用于正常退出程序，它通过调用所有用`atexit`功能注册的功能开始。

`atexit`函数接受一个指向没有参数或返回类型的函数的指针。当调用`exit`函数时，如果该函数成功注册为在程序终止时调用，则返回`0`，否则返回非零值。 `atexit`函数具有以下签名:

```
int atexit(void (*func) (void) );
```

在以反向注册顺序调用用`atexit`函数注册的函数后，`exit`函数关闭所有打开的流，导致它们被刷新。由`tmpfile`功能创建的文件被删除。

`tmpfile`函数是`stdio.h`头的一部分，用于创建临时文件，关闭时或程序正常终止时移除。

接下来，如果传递的状态参数是在`stdlib.h`头中定义的`0`或整数值宏:`EXIT_SUCCESS`，这意味着成功状态必须被传递给宿主环境，因为成功状态的这种实现定义形式被返回给宿主环境，控制权被放弃给宿主环境。

```
/* exit_success.c */#include <stdlib.h>int main ( int argc , char* argv[] )
{
  exit(0);
  /* Normally exit the program
       with success status .
     exit(EXIT_SUCCESS) could 
       also have been used .*/
}/*
$ cc exit_success.c
# compile the program .$ ./a.out 
# execute the program .$ echo $?
# check the exit status of 
#   the last program . 
0
# The exit status of the last
#   program is 0 , which is
#   success .*/
```

如果**传递的状态**参数是在`stdlib.h`头中定义的整数值宏:`EXIT_FAILURE`，则`exit`函数将失败状态的实现定义形式返回给托管环境，控制权将传递给托管环境。

```
/* exit_failure.c */#include <stdlib.h>int main ( int argc , char* argv[] )
{
  exit(EXIT_FAILURE);
  /* Normally exit  the program
       with failure status .*/
}/*
$ cc exit_failure.c
# compile the program .$ ./a.out 
# execute the program .$ echo $?
# check the exit status of 
#   the last program . 
1
# It is 1 , as such 
#  the exist status 
#  is failure .*/
```

对于任何其他的**通过状态**值，返回给主机环境的退出状态是实现定义的。

```
/* exit_status_implementation.c */#include <stdlib.h>int main ( int argc , char* argv[] )
{
  exit(-1);
  /* Normally exit the program ,
       passing to the exit function
       an exit status of -1 .
     An implementation defined , exit
       status is returned to
       the hosing environment .*/
}/*
$ cc exit_status_implementation.c
# compile the program .$ ./a.out 
# execute the program .$ echo $?
# check the exit status of the 
#   last program . 
255
# The exist status is 255 .*/
```

**多次调用**到`exit`函数的行为是未定义的。

*原载于 2020 年 10 月 29 日*[*【https://twiserandom.com】*](https://twiserandom.com/c/stdlib/what-is-the-c-exit-function/)*。*