# 在 Windows 7 上安装 Cygwin

> 原文：<https://medium.com/analytics-vidhya/install-cygwin-on-windows-7-47100f581783?source=collection_archive---------8----------------------->

![](img/4306bd4810f9b0b5dec99e71af1917af.png)

## 在 Windows 机上播放 linux 的一种方法

这是我正在使用的惠普笔记本电脑:

选择开始>控制面板>系统

64 位 Windows 7 专业版 SP1

![](img/7c82dd74e43242b00c84d713e49e4fb6.png)

到[https://www.cygwin.com](https://www.cygwin.com/)点击链接 setup-x86_64.exe

![](img/632c18bfa134c284328c8392b414508d.png)

遵循屏幕上的说明。

![](img/b3cf318b5f461a4a9b5bd71708712f6b.png)![](img/11deb81cf2e1410321ba5b6b105369f8.png)![](img/cfbc826dc46be53a93a1c27c809bf1d4.png)![](img/7dba0104b965db1e5622e4e1ff3d842f.png)![](img/a32005356e04b16d8ef805748f140340.png)![](img/f0f6556fbebd50b1ec6c4d7a0aaabefc.png)![](img/18383874b33e76fc9bce541fe331b053.png)

我干脆选一个滑铁卢大学的镜像网站。

![](img/a4ba75b7370c0626af9fcc9aff17cece.png)![](img/109c3c98b025a18c5022113f1b99c328.png)![](img/6026f6e62f013e75fdbd2b3d78743b0d.png)![](img/9c0578e594c7ca616c777fcca1bb9209.png)![](img/5d0a15e0e2e81a3504c43fef242c126c.png)

双击 Cygwin 图标启动它。

![](img/b8298bd96bd7edf156f6c93d03d1d4c5.png)

要更改字体大小，请右键单击顶部栏并选择选项…

![](img/70cc88460bc75a82a58b51b9603832cb.png)

选取文本>选择…

![](img/edac8202ab7f22059dee26c3f4da217a.png)

我喜欢 16 号的 Segoe UI Mono。

![](img/0d9ba60432beb7217a02833fd0e8fdf1.png)

让我们创建一个简单的 Java 程序来展示在 Cygwin 下的工作。

创建子目录。

```
An Sheng@An-HPLaptop ~
$ java -version
java version "1.8.0_45"
Java(TM) SE Runtime Environment (build 1.8.0_45-b15)
Java HotSpot(TM) 64-Bit Server VM (build 25.45-b02, mixed mode)An Sheng@An-HPLaptop ~
$ pwd
/home/An ShengAn Sheng@An-HPLaptop ~
$ ls -l
total 0An Sheng@An-HPLaptop ~
$ mkdir developerAn Sheng@An-HPLaptop ~
$ ls -l
total 0
drwxr-xr-x+ 1 An Sheng None 0 Jan 11 00:42 developer
An Sheng@An-HPLaptop ~/developer
$ mkdir javaAn Sheng@An-HPLaptop ~/developer
$ cd java/
```

创建 Java 源文件 CChess.java:

```
An Sheng@An-HPLaptop ~/developer/java
$ vi CChess.java
```

![](img/36838d266e72afb85d591d0ae250c763.png)

下面是 CChess.java 的代码:

```
class CChess {
  public static void main(String[] args) {
    System.out.println("Hello, World!");
  }
}
```

编译并运行 CChess.java:

```
An Sheng@An-HPLaptop ~/developer/java
$ javac CChess.javaAn Sheng@An-HPLaptop ~/developer/java
$ ls -l
total 2
-rwxr-xr-x 1 An Sheng None 419 Jan 11 11:53 CChess.class
-rw-r--r-- 1 An Sheng None 107 Jan 11 00:47 CChess.javaAn Sheng@An-HPLaptop ~/developer/java
$ java CChess
Hello, World!An Sheng@An-HPLaptop ~/developer/java
$
```

尝试一些其他的 linux 命令，比如 date、cal、cat 和 wc -l。

![](img/42353343403f21b76c634a3214c40591.png)

```
$ date
Sun, Jan 12, 2020  5:16:01 PMAn Sheng@An-HPLaptop ~
$ cal
    January 2020
Su Mo Tu We Th Fr Sa
          1  2  3  4
 5  6  7  8  9 10 11
12 13 14 15 16 17 18
19 20 21 22 23 24 25
26 27 28 29 30 31An Sheng@An-HPLaptop ~
$ cat developer/java/CChess.java
class CChess {
  public static void main(String[] args) {
    System.out.println("Hello, World!");
  }
}An Sheng@An-HPLaptop ~
$ wc -l developer/java/CChess.java
6 developer/java/CChess.javaAn Sheng@An-HPLaptop ~
$
```