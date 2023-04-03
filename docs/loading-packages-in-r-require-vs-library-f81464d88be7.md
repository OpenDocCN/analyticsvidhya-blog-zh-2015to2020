# 在 R: require()和 library()中加载包

> 原文：<https://medium.com/analytics-vidhya/loading-packages-in-r-require-vs-library-f81464d88be7?source=collection_archive---------25----------------------->

![](img/b0cf74841d0f0bbde11928899ff157e4.png)

[负空格](https://www.pexels.com/@negativespace) @ [像素](https://www.pexels.com)

[R](https://www.r-project.org/) 是为统计计算和图形开发的特殊软件环境。它可以被认为是一种工具，用于从大量数据中分析和提取模式，从而得出结论。

r 语言附带了一系列非常有用的数据分析函数。但是，用户需要根据其领域安装扩展功能的软件包。这些软件包由综合档案网络(CRAN)托管。目前全世界 R 用户写的包有 15604 个。没有必要被数量庞大的软件包所淹没，因为它们为数百个不同领域的数据分析提供了大量的功能，并且它们都不是由一个用户使用的。

**在 R 中安装软件包**

“install.packages()”用于在 r 上安装新的软件包。例如，下面是如何安装软件包“dplyr”。

```
install.packages("dplyr")
```

**加载已安装的软件包**

应该将已安装的包加载到当前的 R 环境中。require()或 library()函数都可以用来将包加载到 R 环境中。

这是如何使用 require()和 library()加载已安装的软件包“dplyr”的:

```
require("dplyr")library("dplyr")
```

两个函数看起来一样，对吧？

**require() vs library():**

尽管它们看起来一样，require()有一个不言而喻的缺点。

如果要加载的包没有安装或者不存在(可能是打印错误？)，library()会产生一个错误，而 require()只提供一个警告。library()提供的错误会阻止当前的 R 脚本进一步运行，而 require()提供的警告则不会。

假设您正在运行一个非常耗时的大型 R 脚本，如果您使用 require()而不是 library()来加载包，该脚本将一直运行，直到它实际调用由于某种原因未能加载的包。相反，如果您使用 library()，它会停止运行脚本，通知您采取必要的措施。

**require()的用法**

require()有这个小缺陷并不意味着它完全没有用。它在许多其他方面也很有用。例如，require()可以用来检查我们试图加载的包是否可用。每当用包名调用 require()时，它都返回一个二进制值，TRUE 或 FALSE，表示包是否可以加载。

```
# Capturing the return value of require(dplyr) to variable xx <- require(dplyr)# If the dplyr package is available, it returns TURE and if it is not, it returns FALSE
```

这在自动化检查和安装 R 脚本中的包的过程中很有用。请参见下面的示例。

```
if(!require(dplyr)) { install.packages("dplyr"); library(dplyr) }
```

这将检查程序包 dplyr 的可用性。如果不是，软件包会自动安装并加载到环境中。

library()和 require()函数在处理 R 中的包时非常有用，用户应该知道如何以及何时使用它们，以确保使用该语言进行数据分析的最大结果。