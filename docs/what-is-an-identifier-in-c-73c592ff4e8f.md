# 什么是 c 语言中的标识符？

> 原文：<https://medium.com/analytics-vidhya/what-is-an-identifier-in-c-73c592ff4e8f?source=collection_archive---------24----------------------->

# 什么是标识符？

一个标识符，用于 ***表示`C`中的一个对象类型*** 。`C`中的对象是内存的一个区域，用于数据存储。

```
int x ; /*
x is an identifier , or the 
  label for an area of storage 
  of type int .*/struct time{
  int hours ;
  int seconds;
};/*
time is an identifier , for an 
  aggregate user defined type ,
  struct .
time is also known as a tag . The
  term tag , is also used to refer ,
  to user defined types : enum , 
  union .
struct time can be used to define ,
  new variables , of type struct
  time . 
variables of types struct time , 
   occupy a region of storage 
   in memory capable of holding
   two integers .*/enum color{red , green , blue }/*
color is an identifier , for a user
  defined type , enum . In this ,
  case the user defined type , is
  an integer . color is also 
  known as a tag . 
enum color , can be used to define 
  new variables of type enum color ,
  which can take integer values .
red , green , blue are integer constants .*/
```

***c 中的*** 对象类型有:

*   整数类型，如 int、short、enum …
*   浮动类型，如 float、double、long double。
*   结构。
*   工会。
*   数组。
*   指针。
*   原子类型。

一个标识符，也用来表示 : 联合、结构和枚举的 ***成员。所以在前面的例子中，`hours`、`seconds`、`red`和`green`也是标识符。***

一个标识符也可以用来表示一个 ***功能*** ，例如:

```
int aFunction(void);/*
aFunction is an identifier . */
```

标识符也可以用来表示: ***标签名称*** 。

```
#include<stdio.h>
void edit(int code){
  if(code == 0) 
    goto success ;
  else
    goto failure ;
 success: // success is an identifier for a label
  printf("sucess\n");
 failure: // failure is an identifier for a label
  printf("failure\n");}int main(void){
  edit(0);
  edit(1);}/*
Output : 
sucess
failure
failure*/
```

最后，标识符 ***可以用来表示***:typedef 名称、宏名称和宏参数。

```
typedef float weight ; /*
weight is an identifier */#define SUM(x,y) (x + y) ;/*
SUM , x , y are identifiers.*/
```

# 标识符的字符和长度

一个标识符是区分大小写的，它 ***可以包含*** 以下字符:`[A-Za-z0-9_]`。它必须以非数字字符开头，因此不能以从`0`到`9`的数字开头。

通用字符名称也可用于标识符中，因此标识符中可使用一个后跟四个十六进制数字的`\u`，或一个后跟八个十六进制数字的`\U`。这个十六进制数字，必须组成一个有效的 [unicode 编码的数字](https://twiserandom.com/python/what-is-a-character-in-python/)。

通用字符名称可以在某些字符不可用的系统上使用。允许的通用字符名称的范围可以在 [C 标准](http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2310.pdf)的附录 D.1 中找到，至于在标识符开头不允许的通用字符名称，可以在 C 标准的附录 D.2 中找到。

```
/*
Example of a valid identifiers*/
int _valid;
int int \u1D00num = 1;/*
Example of an invalid identifiers */
int 1a = 0 ;
```

C 标准没有定义标识符长度的限制，而一些实现可能会设置限制。因此，C 标准对 ***最小长度*** 施加了限制，实现必须支持。

一个实现必须支持的最小字符数，对于外部链接标识符，如全局变量或全局函数，至少是`31`个字符，对于那些具有内部链接的非全局变量或函数，是`63`个字符。

# 标识符的命名空间

标识符可以在以下四个名称空间**之一:**

*   **结构、联合和枚举的标记形成一个命名空间。**
*   **每个结构和联合为其成员形成一个命名空间。**
*   **标签有自己的命名空间。**
*   **所有其他标识符都属于同一个名称空间，它们被称为普通标识符。**

**我们不能在同一个名称空间中定义相同的标识符，除非它们属于不同的作用域。**

```
**struct x{int x;};
enum y{x};
int z;
int func(int z );
/*
The tag x for the struct x , and the 
  tag y for the enum y , belong to 
  the same namespace . 
Each struct and union , form
  a namespace for its members . 
  As such int x; belong to the 
    namespace of the struct x .
Labels form their own namespace .
All others identifiers belong to the 
  same name space . As such , x , z ,
  func , and z , belong to the same 
  namespace , but they might have 
  different scope . Scope is visibility .*/**
```

# **标识符的范围**

**标识符的作用域是该标识符的 ***可见性*** 。标识符可以属于四个范围:**

*****函数原型范围*** ，是针对每个函数原型内部定义的参数。没有必要为函数原型中定义的参数提供标识符。**

```
**int x ; 
int sum(int x , int y );/*
x , and x belong to the same name 
  space , the ordinary name space , 
  but they have different scopes .
In the sum function prototype declaration , 
  we have two parameters , which have
  the sum function prototype scope ,
  they are : x and y .*/**
```

**标签里面声明的函数，有 ***函数作用域*** 。每个功能，都有自己的功能范围。**

*****块范围*** ，由花括号定义，它以`{`开始，以`}`结束。属于功能定义块范围的功能参数。**

```
**#include<stdio.h>void dimension(int x , int y ){
  for( int i = 0 ; i< 10 ; i++){
    printf("%d",i);}
  printf("\n");
  printf("%d,%d\n", x , y);}/*
x , y , belong to the block scope ,
  following the parameter list ,
  of the dimension function , 
  and are only visible inside this 
  block scope .
i , belong to the block scope , 
  following the for loop clauses , 
  and is only visible inside this 
block scope.*/int main(void){
  dimension(0,1);}/*
Output :
0123456789
0,1*/**
```

*****文件作用域*** ，是所有其他的东西，所以任何没有在块作用域、函数作用域或函数原型作用域中定义的东西。文件范围是全局范围。**

```
**int x = 1 ;/*
x belong to the file scope .*/**
```

**一个标识符是 ***可见的*** ，从它被声明的那一刻起，直到它的作用域结束。一个文件范围的终点，是[翻译单元](https://twiserandom.com/c/the-compilation-process-of-a-c-source-file/)。嵌套块隐藏了外部块中定义的标识符的可见性。**

```
**#include<stdio.h>int a = -1;void options(int a){
  printf("a : %d\n",a);
  if(a == 0){
    int a = 1 ; 
    printf("a : %d\n",a);}
  printf("a : %d\n",a);}int main(void){
  printf("a : %d\n",a);
  options(0);
  int z = -1 , t = z;
  printf("z : %d , t : %d\n", z , t);}/*
Output :
a : -1
a : 0
a : 1
a : 0
z : -1 , t : -1**
```

***原载于 2020 年 11 月 19 日*[*https://twiserandom.com*](https://twiserandom.com/c/what-is-an-identifier-in-c/)*。***