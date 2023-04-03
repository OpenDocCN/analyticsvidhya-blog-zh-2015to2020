# 使用 Golang 的文件管理工具

> 原文：<https://medium.com/analytics-vidhya/file-management-tool-using-golang-9858c9278cf3?source=collection_archive---------9----------------------->

您是否曾经遇到过这样的情况:您需要重命名一堆到处都是的文件？或者你是否遇到过这样的情况:你想选择性地复制文件，但它们与其他一些文件在一起，你不能简单地选择所有文件？

![](img/1fe95182b3d87d0b3105a30c7eafecf4.png)

Sebastian Herrmann 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在我们的生活或工作中，我们可能需要执行这些耗费大量时间的乏味任务，最好使用代码行来自动完成这些杂务。

在这篇[文章中，](/swlh/building-cli-based-file-renaming-tool-with-golang-e3c6a16eedf6)我分享了我们如何使用 Golang 构建一个基于 CLI 的文件重命名工具，在这篇文章中，我们将进一步扩展功能到**复制、移动和压缩文件**，同时我们将研究 Golang 标准库如何帮助我们实现我们想要的。

# 复制功能

对于复制功能，我们需要一个源文件所在的源路径和一个目标路径。请注意，如果目标路径中存在现有文件，则现有文件将被替换。

复制功能

1.  第 7 行:`os.Open()`在给定的路径下打开文件进行读取
2.  第 15 行:`os.Stat()`检查目标文件夹是否存在，如果不存在，它将进入`if`循环
3.  第 16 行:`os.MkdirAll()`创建包含任何必要父目录的目录。`os.ModePerm`是目录的文件模式和权限，相当于`Chmod 777`
4.  第 21 行:`os.Create()`在给定的路径下创建一个文件。如果路径中存在文件，它将被截断
5.  第 27 行:`io.Copy()`将文件从源文件复制到目的文件，直到源文件到达 EOF

# 移动功能

移动功能

与复制功能类似，我们将检查目标目录是否存在，如果不存在，我们将创建目标目录及其父目录(如果需要)。

1.  第 14 行:`os.Rename()`重命名源文件的整个路径，这实际上将文件移动到一个新的位置

# 压缩功能

有一个 ZIP 包`archive/zip`提供了对 ZIP 存档的读写支持。在下面的代码中，我们将传入 ZIP 文件的名称以及要归档的文件。

压缩功能

1.  第 9 行:`os.Create()`用我们传入的给定名称创建一个文件
2.  第 15 行:`zip.NewWriter()`实例化一个 zip 文件编写器
3.  第 19–23 行:由于文件夹可以是用户键入的输入，我们需要通过`os.Chdir()`将当前工作目录更改到指定的文件夹。然后我们在`os.Getwd()`中返回对应于当前工作目录的根路径名
4.  第 29–34 行:遍历每个文件，我们使用`filepath.Rel()`来获取文件相对于基路径的路径，这样当我们以后想要放入 zip 存档时，可以保留文件的结构。有了相对路径，我们再在`filepath.Join()`中形成一条路径
5.  第 35 行:形成路径后，我们调用函数`addFileToArchive()`将文件放入 zip 中
6.  第 59 行:`fileToZip.Stat()`返回描述文件的`FileInfo`结构。
7.  第 64–70 行:使用第 59 行的`FileInfo`，`zip.FileInfoHeader()`创建一个部分填充的`FileHeader`，它描述 zip 中的文件。我们在第 69 和 70 行指定了名称和压缩方法(deflate)
8.  第 72 行:`zipWriter.CreateHeader()`将文件添加到 zip 中，并返回一个`writer`，它将用于写入文件的内容
9.  第 76 行:`os.Copy()`将内容从源文件复制到 zip 文件中

这就是文件管理工具的核心功能。对于完整的源代码，你可以看看[https://github.com/wilsontwm/filezy](https://github.com/wilsontwm/filezy)我已经建立了一个基于 CLI 的文件管理工具。

谢谢，干杯！