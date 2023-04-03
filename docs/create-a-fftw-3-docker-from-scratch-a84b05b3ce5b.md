# 从头开始创建一个 FFTW 3 码头

> 原文：<https://medium.com/analytics-vidhya/create-a-fftw-3-docker-from-scratch-a84b05b3ce5b?source=collection_archive---------32----------------------->

## 开始为集装箱化的目标塑造形象吧！

![](img/7a21bdf9a848b09a89d98df9ee0f763f.png)

照片由 [Ishant Mishra](https://unsplash.com/@ishant_mishra54?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

> *TL；DR:可以从* [*这里*](https://hub.docker.com/r/cudachen/fftw-docker) *拉图。*
> 
> 点击[此](/analytics-vidhya/create-a-fftw-3-docker-from-scratch-a84b05b3ce5b?source=friends_link&sk=b90be6bcf6964532860a9dd0fbead80d)通过付费墙。

# 动机

FFTW 的速度相当惊人，但安装有点复杂。更重要的是，我想尝试从 GitHub repo 而不是 zip 文件中构建 FFTW 源代码，并且我想要一个配置良好的移动环境。因此，我创建了这个项目并学习了一些关于 Docker 的基础知识。

# Dockerfile 文件的内容

长话短说，你可以在这里看到整个 docker file。

我将概述这个文档的一些重要部分。

# 基础图像

在第一行，你可以看到我选择了 Ubuntu 的基本图像。虽然有些人认为使用这样的图像会导致巨大的图像尺寸，但我认为我应该使用这种图像，因为它包含标准的`glibc`，这对 FFTW 来说至关重要。

# 为什么 FFTW 要求 OCaml 从零开始构建？

在第 15~18 行，你可以看到我下载了一些包，包括 OCaml。原因是作者使用 OCaml 生成“小代码”，这些小代码是硬编码的代码，以便在特定条件下达到最佳性能。[1]

如果你曾经上过编译器的课程，你会意识到用 C 实现一个小型编译器不仅繁琐，而且需要维护 AST(抽象语法树)[2]的知识，即使你用`lex`和`yacc`做功课。所以班上一些很酷的家伙会毫不费力地使用其他语言来创建 AST，如 Haskell 和 OCaml，从而完美地完成作业。你猜怎么着？作者选择后者来创建那些“小代码”，让 FFTW 生成优化的 C 代码。

那我为什么要下载其他四个包呢？因为当我试图在 Ubuntu Docker 中构建 FFTW 时，容器总是抱怨没有这些包就无法编译。那一刻，我意识到，一个基础映像通常会为了缩小映像大小而丢弃一些包。

那么如何解决这个问题呢？很简单。您只需读取错误，记下错误，在编译前添加缺失的包，然后循环，直到构建没有错误(真的)。

# 最后，为什么我们需要 ocaml 数字库？

很简单。在 FFTW GitHub repo 的 README 中提到，OCaml 在 OCaml 4.06.0 中删除了 ocaml Num 库(2017 年 11 月 3 日)。[3]

有趣的是 Ubuntu 18.04 在 APT 上没有这样的包。因此，我手动下载并从源代码编译它。

# 想到了这个项目

完成这个项目后，我不仅意识到如何从零开始编译 FFTW。更重要的是，我做了一个很好的实践来创建一个符合我需要的 Docker 图像，并且我有机会深入挖掘这个使用其他语言来解决 c 语言陷阱的杰作。

最后，希望你们喜欢这篇文章！

# 参考

[1][http://www.fftw.org/fftw3_doc/Generating-your-own-code.html](http://www.fftw.org/fftw3_doc/Generating-your-own-code.html)

[2]https://en.wikipedia.org/wiki/Abstract_syntax_tree

[3]https://github.com/FFTW/fftw3/blob/master/README

*原载于 2020 年 5 月 18 日*[*https://cuda-Chen . github . io*](https://cuda-chen.github.io/devops/2020/05/18/Create-a-FFTW-3-Docker-from-Scratch.html)*。*

> 如果你有什么想法和问题要分享，请联系我[**clh 960524【at】Gmail . com**](http://clh960524@gmail.com/)。另外，你可以查看我的 [GitHub 库](https://github.com/Cuda-Chen)的其他作品。如果你像我一样对机器学习、图像处理和并行计算充满热情，欢迎在 LinkedIn 上加我。