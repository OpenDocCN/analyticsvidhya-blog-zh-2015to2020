# 安装和使用 GNU Parallel 的简单教程

> 原文：<https://medium.com/analytics-vidhya/simple-tutorial-to-install-use-gnu-parallel-79251120d618?source=collection_archive---------0----------------------->

## 如何在 Centos / RHEL 服务器上安装 GNU Parallel？

![](img/7639d00607730401cdac6a8e20790a28.png)

图片来源:wikipedia.org

你在这里是因为你知道 GNU 并行的力量..以下是安装和使用它的简单步骤。

下载最新版本的 GNU 并行

```
$ wget [http://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2](http://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2)
```

解压它，它会创建一个新的文件夹 **parallel-yyyymmdd**

```
$ sudo tar xjf parallel-latest.tar.bz2
```

更改到新文件夹

```
$ cd **parallel-yyyymmdd**
```

在你的机器上构建软件

```
$ sudo ./configure && make
```

安装它

```
$ sudo make install
```

切换到主目录

```
$ cd
```

**如何测试水货？**

```
$ parallel
```

按 Ctrl D 或 Ctrl C 退出并行。

**用例**

我的常见用例是使用 Parallels 压缩一个大文件。4000 万行(8 GB)的文件被压缩为 400MB 的 bz2 文件。

```
$ cat largefile.csv | /usr/local/bin/parallel --pipe -k bzip2 --best > largefile.bz2
```