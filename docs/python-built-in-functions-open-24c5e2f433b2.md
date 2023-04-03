# Python 内置函数:open()

> 原文：<https://medium.com/analytics-vidhya/python-built-in-functions-open-24c5e2f433b2?source=collection_archive---------31----------------------->

# Python 中文件处理的入门

![](img/8cb6e3b13cd43e132d76d08db862714f.png)

**open()** 将打开一个文件并返回一个 file 对象:

```
myhandle = open('demo')
print(myhandle.read())
```

这返回了一些披头士的歌词:

```
Yesterday
All my troubles seemed so far away
Now it looks as though they're here to stay
```

如果您想确定您正在读取文件，请使用“r”模式:

```
myhandle = open('demo', 'r')
print(myhandle.read())
```

事实上， **open(filename，mode)** 是一种公认的编码方式，可以避免别人在阅读你的代码时产生混淆，默认为“r”。

如果你准备好照看你自己的代码块，直接进入文件当然会工作，但是通常使用带有语句的**:**

```
with open('demo','r') as my_handle:
    print(my_handle.read())
```

这给出了相同的结果。**与**将在文件块结束时自动关闭文件。如果你想读写现有的文件呢？在模式中添加一个“+”:

```
with open('demo', 'r+') as my_handle:
    my_handle.write('\nOh, I believe in yesterday')
    print(my_handle.read())
```

什么？不显示文本？如果你检查“演示”,你会看到文本在那里。python 处理文件位置的方式是使用光标的概念，光标停留在文件最后一次被读写的地方。使用' tell()'找出您的光标在哪里:

```
with open('demo','r+') as my_handle: print('cursor is now at:')
    print(my_handle.tell()) *# cursor at the beginning*
    print(my_handle.read()) *# reads the text and leaves the cursor at the end*
    print('read text, cursor is now at:')
    print(my_handle.tell()) *# cursor at the end of file* my_handle.write('\nOh, I believe in yesterday')
    print('added text, cursor is now at:')
    print(my_handle.tell()) *# writes the above text and leaves the cursor at the end* my_handle.seek(0) *# rewind cursor to beginning*
    print('rewound cursor, now at:')
    print(my_handle.tell()) *# check where we are*
    print(my_handle.read()) *# read all the text again*
    print('just read added text, cursor is now at:')
    print(my_handle.tell()) *# cursor is at the end* my_handle.seek(20) *# demo - place the cursor 20 chars in*
    print('just moved 20 characters up, now at:')
    print(my_handle.tell()) *# check where the cursor is*
    print(my_handle.read()) *# read from here*
```

这将输出:

```
cursor is now at:
0
Yesterday
All my troubles seemed so far away
Now it looks as though they're here to stay
read text, cursor is now at:
88
added text, cursor is now at:
115
rewound cursor, now at:
0
Yesterday
All my troubles seemed so far away
Now it looks as though they're here to stay
Oh, I believe in yesterday
just read added text, cursor is now at:
115
just moved 20 characters up, now at:
20
ubles seemed so far away
Now it looks as though they're here to stay
Oh, I believe in yesterday
```

我还在代码中添加了一些注释，只是为了弄清楚到底发生了什么。

# r+和 w+有什么区别？

那么如果 **r+** 让你读写，而 **w+** 让你读写，那有什么区别呢？那么，如果“演示”文件在开始时丢失了怎么办？ **r+** 将出错退出。要解决这个问题，将 **r+** 模式改为 **w+** 。这仍然会截断文件，但如果它不存在，也会创建它。

将 r+和 w+放在上下文中，有一组可用的模式:

```
+--------------------------+---+----+---+----+---+----+---+----+
|                          | r | r+ | w | w+ | x | x+ | a | a+ |
+--------------------------+---+----+---+----+---+----+---+----+
| read                     | + | +  |   | +  |   | +  |   | +  |
| write                    |   | +  | + | +  | + | +  | + | +  |
| write after seek         |   | +  | + | +  | + | +  |   |    |
| create                   |   |    | + | +  | + | +  | + | +  |
| truncate                 |   |    | + | +  |   |    |   |    |
| cursor position at start | + | +  | + | +  | + | +  |   |    |
| cursor position at end   |   |    |   |    |   |    | + | +  |
| notes                    | 1 | 2  | 3 | 4  | 5 | 6  | 7 | 8  |
+--------------------------+---+----+---+----+---+----+---+----+
```

…但是这里需要一些解释:

1.  “读取”将打开一个现有文件供*读取*，但如果该文件不存在，将给出一个 **FileNotFoundError** 。
2.  “read+”将打开一个现有文件供*读*和*写*，但如果该文件不存在，将给出一个 **FileNotFoundError** 。
3.  “write”将打开一个现有文件供*写入*并在这样做之前截断它，*或者它将创建一个不存在的文件。*
4.  “write+”将打开一个现有文件供*读取*和*写入*，并在执行此操作之前将其截断，*或者如果该文件不存在，将创建一个文件。*
5.  “x”将创建一个新的写入文件，但如果它已经存在，将给出一个**filexistserror**。文件不能被截断，因为它的开头是空的。
6.  x+'将创建一个新的读写文件，但如果它已经存在，将给出一个**filexistserror**。文件不能被截断，因为它的开头是空的。
7.  “append”将打开一个现有文件并追加到末尾，如果它不存在，则创建一个新文件。
8.  ' append+'将打开一个现有的文件进行读取并追加到末尾，如果它不存在，则创建一个新文件。

(有什么用？掌声很好，或者评论/问题…)