# python 中的酸洗和反酸洗解释

> 原文：<https://medium.com/analytics-vidhya/pickling-and-unpickling-in-python-explained-be546e604aae?source=collection_archive---------4----------------------->

![](img/5749be8dc9db4f357682eca64e9e7c2b.png)

由[麦克斯韦·纳尔逊](https://unsplash.com/@maxcodes?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

Pickling 允许您序列化和反序列化 Python 对象结构。简而言之，Pickling 是一种将 python 对象转换成字符流的方法，这样这个字符流就包含了在另一个 python 脚本中重建对象所需的所有信息。

为了成功地重建对象，标记的字节流包含给拆包器的指令，以重建原始对象结构以及指令操作数，这有助于填充对象结构。

# Pickle 协议:

目前有 6 种不同的方案可用于酸洗。使用的协议越高，读取产生的 Pickle 所需的 Python 版本就越新。了解更多关于 [Pickle 协议](https://docs.python.org/3/library/pickle.html)的信息。

泡菜主要有两种方法。第一个是`dump`，它将一个对象转储到一个文件对象，第二个是`load`，它从一个文件对象加载一个对象。

```
pickle.dump(object, file_obj, protocol)
```

这个函数有三个参数-

1.  要序列化的 Python 对象。
2.  必须在其中存储序列化 python 对象的文件对象。
3.  协议(如果未指定协议，则使用协议 0。如果协议被指定为负值或`HIGHEST_PROTOCOL`，将使用可用的最高协议版本。)

所以让我们继续就事论事:

*   首先，我们必须通过以下命令导入它:

```
import pickle
```

下面是酸洗和拆线的例子-

# 酸洗

```
import pickledef pickle_data():
    data = {
                'name': 'Prashant',
                'profession': 'Software Engineer',
                'country': 'India'
        }
    filename = 'PersonalInfo'
    outfile = open(filename, 'wb')
    pickle.dump(data,outfile)
    outfile.close()pickle_data()
```

我们使用`open`函数打开文件，第一个参数应该是你的文件名，第二个参数是`wb`，`wb`是指以二进制方式编写。这意味着数据将以字节对象的形式写入。

# 拆线

```
import pickledef unpickling_data():
    file = open(filename,'rb')
    new_data = pickle.load(file)
    file.close()
    return new_data print(unpickling_data())
```

这里`r`代表读取模式，`b`代表二进制模式。你将会读到一个二进制文件。
拆包功能的输出与我们腌制的相同。

```
{'name': 'Prashant', 'profession': 'Software Engineer', 'country': 'India'}
```

# 有什么可以腌制的？

可以对以下数据类型进行酸洗:

*   布尔人，
*   整数，
*   浮动，
*   复数，
*   (普通和 Unicode)字符串，
*   元组，
*   列表，
*   包含可拾取对象的集合和字典。
*   在模块顶层定义的内置函数
*   在模块顶层定义的类

# 酸洗的常用案例

1.  将程序的状态数据保存到磁盘上，这样当重新启动时，它可以从停止的地方继续运行(持久化)
2.  在多核或分布式系统中通过 TCP 连接发送 Python 数据(编组)
3.  将 Python 对象存储在数据库中

# 酸洗的危险

Pickle 模块的文档说明:

> Pickle 模块无法抵御错误或恶意构建的数据。不要从不受信任或未经验证的来源提取数据。

# 泡菜的安全实施

*   泡菜不应该在陌生人之间使用。
*   确保交换 Pickle 的各方具有加密的网络连接。
*   在不安全连接的情况下，Pickle 中的任何更改都可以通过使用加密签名来验证。Pickle 可以在传输之前进行签名，它的签名可以在加载到接收端之前进行验证。

我希望您现在对 python 中的酸洗/反酸洗有了一个相当好的理解。

如果你有什么建议，请在评论中告诉我。

# 参考

 [## pickle - Python 对象序列化- Python 3.9.1 文档

### 编辑描述

docs.python.org](https://docs.python.org/3.8/library/pickle.html) [](https://stackoverflow.com/questions/7501947/understanding-pickling-in-python) [## 理解 Python 中的 Pickling

### 我最近接到一个任务，需要把一个字典(其中每个键都指向一个列表)以腌泡的形式…

stackoverflow.com](https://stackoverflow.com/questions/7501947/understanding-pickling-in-python)