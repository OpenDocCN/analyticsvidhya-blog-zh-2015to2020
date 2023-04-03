# C 语言中的指针是什么，它们是如何工作的？

> 原文：<https://medium.com/analytics-vidhya/what-are-pointers-in-c-and-how-do-they-work-1ecf297d94c5?source=collection_archive---------15----------------------->

## 指针是引用存储单元的数据值。

假设我们想创建一些指针，指向一些包含整数的存储块，用 C 语言实现这一点最简单的方法如下:

首先，我们必须在。c 文件:
**typedef int * pointerto integer；**

```
Here is how the .c file should look at this point:#include <stdio.h>
**typedef int**…
```