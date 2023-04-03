# 静态库和动态库的区别。

> 原文：<https://medium.com/analytics-vidhya/differences-between-static-and-dynamic-libraries-17dd5e41da96?source=collection_archive---------22----------------------->

![](img/0e486129126f2be3aff01064da4e4bd0.png)

# 为什么使用图书馆？

通常情况下，可以由多个应用程序共享的 C 函数/C++类和方法是从应用程序的源代码中分离出来的，经过编译并捆绑到一个库中。C 标准库和 C++ STL 是可以与您的代码链接的共享组件的例子。好处是不需要声明每个目标文件…