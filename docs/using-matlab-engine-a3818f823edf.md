# 使用 Matlab 引擎

> 原文：<https://medium.com/analytics-vidhya/using-matlab-engine-a3818f823edf?source=collection_archive---------13----------------------->

![](img/57b723b13fd04d47da9107a39a664da9.png)

在做一个项目时，我意识到如果我们能把 MATLAB 和 Python 结合起来，那么处理图像将变得很容易。所以，我开始在互联网上搜索，看看我们是否可以做到这一点，在那里我知道这是可能的使用 MATLAB API 命名的 MATLAB 引擎。

在这篇博客中，我将通过指导你安装和使用这个 API 的过程来帮助你。

## 先决条件

1.  MATLAB 版本应该在 R2014b 以上。
2.  检查你是否安装了与你的 MATLAB 兼容的 Python 版本。
3.  放 Python 的路径，如果之前没有做。
4.  找到 MATLAB 文件夹的路径。你只需在你的 MATLAB 的命令窗口中写下 *matlabroot* 就可以做到这一点。复制此路径。

## 安装 API 的步骤

**在 Windows 上安装**

1.  打开命令提示符
2.  编写命令:*CD " matlabroot \ extern \ engines \ python "*。注意:使用 matlabroot 地址而不是 matlabroot。
3.  下一个类型: *python setup.py install*

**在 Mac 或 Linux 上安装**

1.  开放终端
2.  编写命令:*CD " matlabroot/extern/engines/python "*。注意:使用 matlabroot 地址而不是 matlabroot。
3.  下一个类型: *python setup.py 安装*

**在 MATLAB 上安装**

1.  转到命令窗口，键入命令: *cd (fullfile(matlabroot，' extern '，' engines '，' python '))。*注意:使用 matlabroot 地址代替 matlabroot。
2.  下一个类型:*系统(' python setup.py install')*

# 启动 MATLAB 引擎

1.  导入 MATLAB 引擎:*导入 matlab.engine*
2.  声明一个变量来启动引擎。*eng = MATLAB . engine . start _ MATLAB()*
3.  使用以下命令向 MATLAB 发送图像。*工程名称 matlab 文件(图片)*

提示:把你的 matlab 文件转换成一个函数，这个函数以图像作为参数。不要为此使用在线 python 编译器。我会推荐视觉工作室和木星笔记本。

所以，这都是我这边的。希望对你有帮助。我写这篇博客是为了让其他人不要面对我在使用 MATLAB 引擎时遇到的困难。