# 我们来看看 Python 中的 if __name__ == '__main__ 是什么

> 原文：<https://medium.com/analytics-vidhya/lets-see-what-is-if-name-main-in-python-9bec8ff3a80a?source=collection_archive---------8----------------------->

在执行代码之前，Python 解释器读取源文件并定义一些特殊变量/全局变量。
如果 python 解释器将该模块(源文件)作为主程序运行，它将特殊的 __name__ 变量设置为值 **"__main__"** 。

如果该文件是从另一个模块导入的，__name__ 将被设置为**模块的名称。**模块名可作为 __name__ 全局变量的值。
模块是包含 Python 定义和语句的文件。文件名是带后缀的模块名。py 已附加。当我们将文件作为命令执行给 python 解释器时，

```
# Python program to execute # main directly**print** "Always executed"**if** __name__ **==** "__main__":
   print "Executed when invoked directly"**else**:
   **print** "Executed when imported"
```

*   缩进级别为 0 [Block 1]的所有代码都会被执行。定义的函数和类是定义好了的，但是它们的代码都没有运行。
*   在这里，就像我们直接执行 script . py _ _ name _ _ 一样，变量将是 **__main__** 。因此，只有当该模块是程序的入口点时，if 块[Block 2]中的代码才会运行。
*   因此，您可以通过测试 __name__ variable 来测试您的脚本是直接运行还是由其他程序导入。
*   如果脚本被其他模块导入，那么 **__name__** 将是模块名。

**我们为什么需要它？**

例如，我们正在开发一个脚本，该脚本旨在用作一个模块:

```
# Python program to execute# function directly**def** my_function():
    print "I am inside function"# We can test function by calling it.my_function()
```

现在，如果我们想通过导入来使用那个模块，我们必须注释掉我们的调用。最好的方法是使用下面的代码，而不是那种方法:

```
# Python program to use# main for function call.**if** __name__ **==** "__main__":
    my_function()**import** myscripy
myscript.my_function()
```

**优点:**

1.  每个 Python 模块都定义了它的 __name__，如果这是“__main__”，则意味着用户正在独立运行该模块，我们可以执行相应的适当操作。
2.  如果将此脚本作为模块导入到另一个脚本中，则 __name__ 将设置为脚本/模块的名称。
3.  Python 文件既可以作为可重用模块，也可以作为独立程序。
4.  if __name__ == "main ":用于执行某些代码**只有**文件是直接运行的，而不是导入的。