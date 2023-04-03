# 通过 python 使用 Tesseract

> 原文：<https://medium.com/analytics-vidhya/using-tesseract-with-python-1cadbe37e756?source=collection_archive---------7----------------------->

![](img/0bc944af0d8cc0b27c0c2b9d9ae4ed45.png)

Tesseract-ocr 是用于各种操作系统的光学字符识别引擎。它是自由软件，在 Apache 许可下发布。并于 2005 年开源，自 2006 年以来一直由谷歌赞助。

这个引擎有它的二进制文件，一个 CLI 程序，它的原始 API 是用 C++制作的。其他编程语言有自己的库，这些库充当 CLI 程序或 C++上原始 tesseract API 的包装器。

在本文中，我们将讨论在 python 中使用 tesseract 的不同选项(它们的安装、执行不同的 API 端)以及在使用过程中遇到的一些重要问题。

> 在 python 中有两个最流行的选项:pytesseract 和 tesserocr。
> 
> tesserocr 是围绕 Tesseract C++ API 的 python 包装器。另一方面，pytesseract 是 tesserac-ocr CLI 程序的包装器。
> 
> 正如你可以猜到的，tesserocr 提供了更多的灵活性和对 tesseract 的控制。Tesserocr 具有多处理能力，在实际应用中比 PyTesseract 快得多。

根据您的使用情况，您可能需要旧的软件包 3.05 或最新的宇宙魔方版本 4+。例如，tesseract 4.0 不提供已识别文本的字体信息，因此您需要旧版本来收集这些信息。

## 通用设置:

为了尝试这些例子，您需要安装 [anaconda](https://www.anaconda.com/products/individual) 来管理包和环境变量。
安装完成后，创建一个新的 python 安装环境，这是所使用的 tesseract 版本所要求的(注意:对于我们的示例，使用 python 3.7)，并在每次运行任何示例时激活该环境:

```
**create environment:** conda create -n tesseractEnvironment python=3.7
**activate environment:** conda activate tesseractEnvironment
```

# 宇宙魔方

要使用任何 tesseract python 包装器，我们需要首先安装 tesseract-ocr。要在 Debian/Ubuntu 上安装 tesseract:

```
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

***注意:*** *以上命令行将安装最新可用版本的 tesseract-ocr，即 tesseract 4。如果您想使用 3.05 这样的早期版本，您必须手动编译 tesseract 存储库，如果您想提取文本的字体相关信息，这可能是必要的，如果手动编译 tesseract 3.05 太复杂，那么我们将进一步讨论一个解决方案来完成设置。*

对于 windows 安装检查:[https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

宇宙魔方具有各种类型的预训练模型，这些模型在速度和准确性之间具有折衷，如果要使用的话，这些模型将与宇宙魔方-ocr 路径一起提供。这些模型包括:
1。tessdata(适用于传统宇宙魔方，即 3.05)
2。tessdata_best(最新版本)
3。tessdata_fast(最新版本)
根据您的使用案例下载 tessdata 预训练模型。默认情况下，你不需要担心 tessdata 的设置，因为默认的 tessdata 可用于 tesserac t-ocr【https://github.com/tesseract-ocr/tesseract-ocr.github.io】参考:[了解更多信息](https://github.com/tesseract-ocr/tesseract-ocr.github.io)

现在，我们将先看看 PyTesseract 的一些 API，然后再看看 Tesserocr
本文将涵盖以下部分的主题:
1 .设置
2。暴露的 API 描述
3。代码片段示例

# 宇宙魔方

**设置:** 用于在您的环境中安装 Pytesseract:

```
conda install -c conda-forge pytesseract
```

如果 tesseract 不在您的路径中，您必须在代码中使用 pytessera CT . pytessera CT . tessera CT _ cmd 变量提供安装路径。

## API 描述

pytesseract 公开了以下流行的 API:
1。pytesserac t . image _ to _ string(image)返回图像中识别的文本
2。pytesserac t . image _ to _ boxes(image)每个识别单词的包围盒
3。pytesserac t . image _ to _ data(image)与已识别单词的尺寸、线条、段落信息相关的数据
4。pytesseract.image_to_osd(image)置信度和其他页面相关信息
5。pytesserac t . image _ to _ pdf _ or _ hocr(image，extension='pdf' /'hocr ')图像或 xml 格式信息的 pdf 格式

## 代码片段示例

运行下面的代码片段来更好地了解 API
*注意:如果下面的代码片段不起作用，请在 tesseract_cmd 变量中提供 tesseract 可执行文件的路径。*

# TesserOCR

安装最新的稳定版本，但如果你想提取字体信息，你需要使用 tesserocr v2.3.1，因为它与 tesseract3.05 兼容。

## 设置

在您的环境中安装 tesserocr:

Debian/Ubuntu:

```
**tesserocr latest:** conda install -c conda-forge tesserocr
**tesserocr v2.3.1:** conda install -c mcs07 tesserocr 
or [https://anaconda.org/mcs07/tesserocr/2.3.1/download/linux-64/tesserocr-2.3.1-py36h6bb024c_0.tar.bz2](https://anaconda.org/mcs07/tesserocr/2.3.1/download/linux-64/tesserocr-2.3.1-py36h6bb024c_0.tar.bz2)
```

窗口:

```
**tesserocr latest:** conda install -c simonflueckiger tesserocr
**tesserocr v2.3.1:** conda install -c simonflueckiger/label/tesseract-3.5.2 tesserocr
```

## API 描述

Tesserocr 有以下有用的 API:
1。tesse rocr . image _ to _ text(image)
2。PyTessBaseAPI()getutf 8 text()
3。PyTessBaseAPI()。MeanTextConf()
4。PyTessBaseAPI()。DetectOrientationScript()
上面的 API 非常简单，通过文档[](https://github.com/sirfz/tesserocr)**可以很容易地理解这些示例。**

*我要研究的课题是如何提取字体数据。通过尝试下面的脚本，您可以得到一个很好的想法。
*注意:如果您想使用某个特定的预训练模型，请提供 path 变量中正在使用的 tessdata 路径。**

*要检查所有的 tessera CT c++ API，请查看:[https://tesseract-ocr.github.io/](https://tesseract-ocr.github.io/)。这些也可以与 tesserocr 一起使用。*

*我们已经到了这个环节的尾声。希望你喜欢并发现这篇文章有用！！*