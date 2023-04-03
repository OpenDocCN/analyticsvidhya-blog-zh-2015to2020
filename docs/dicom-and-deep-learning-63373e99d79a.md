# 提取 DICOM 图像仅用于深度学习

> 原文：<https://medium.com/analytics-vidhya/dicom-and-deep-learning-63373e99d79a?source=collection_archive---------2----------------------->

# 什么是 DICOM？

DICOM 或医学数字成像和通信是来源于不同模式的图像文件，它是传输、存储、检索、打印、处理和显示医学成像信息的国际标准。但是，DICOM 将信息分组到数据集中，这意味着图像文件包含病人信息 ID、出生日期、年龄、性别和其他有关诊断的信息，所有这些都在图像内，如图所示的医学图像的主要组成部分。

![](img/a70aa9e7baaf9fd39ac7451926157134.png)

**医学图像组件**

*   像素深度是用于对每个像素的信息进行编码的位数。例如，一个 8 位栅格可以有 256 个唯一值，范围从 0 到 255。
*   光度解释指定了如何将像素数据解释为单色或彩色图像的正确图像显示。为了指定颜色信息是否存储在图像像素值中，我们引入了每像素样本的概念，也称为(通道数)。
*   元数据是描述图像的信息(即患者 ID、图像日期)。
*   像素数据是存储像素数值的部分。所有组件都是必不可少的，但在我们的范围内，像素深度和像素数据。据我所知，超声图像转换成另一种格式不成问题，但我们必须考虑图像的深度，因为我们不能将 16 位 DICOM 图像转换成 8 位 JPEG 或 PNG，这可能会破坏图像质量和图像特征。像素数据，我们将把它输入网络的数据。

有关 DICOM 格式的更多信息，请访问:

[](http://www.ijri.org/article.asp?issn=0971-3026;year=2012;volume=22;issue=1;spage=4;epage=13;aulast=Varma) [## 管理 DICOM 图像:放射科医师的提示和技巧

### 年份:2012 |卷:22 |期:1 |页:4–13 管理 DICOM 图像:放射科医师的提示和技巧…

www.ijri.org](http://www.ijri.org/article.asp?issn=0971-3026;year=2012;volume=22;issue=1;spage=4;epage=13;aulast=Varma) 

# 为什么只提取图像？

正如我们所看到的，DICOM 格式包含大量信息，有时我们只需要图像，因为这是患者的私人信息，或者我们需要缩小图像的大小，即使信息没有图像本身大，所以我们必须删除图像的元数据，因为我们不允许非医疗人员或工程师查看患者的数据，或者我们不需要冒数据被错误暴露给任何人的风险。主要是我们只需要图像 prat(像素阵列)没有其他任何东西。

# 如何只提取图像？

大多数情况下，我们必须准备数据集来读取所有图像，并将其存储为一个列表，以将其提供给网络，这个过程可能各不相同，就像文件的组织方式一样。

> 开始从目录中收集数据

注意:文件必须按照以下方式组织

我的目录/

|
├──未名 _ 154159/
└──im-0005–0001 . DCM

└──im-0005–0002 . DCM
|
├──未名 _ 136281/
|└──im-0001–0001 . DCM
|
├──未名 _ 190381/
|└──im-0002–0001 . DCM
|
├──未名 _ 102430/
|└──im-0001–002 . DCM【T10
。
。

```
import pydicom as di 
import os
from os import listdirPathDicom = "The/path/to/DICOM/floders"
DCMFiles = [] 
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            DCMFiles.append(os.path.join(dirName,filename))print("Number of (.dcm) files =", len(DCMFiles))
```

现在我们有了包含。dcm 文件，然后我们需要删除所有的数据，只提取像素阵列(图像本身)。

```
import pydicom as diImages1 = []
for k in DCMFiles:
    Images = di.read_file(k,force=True)
    Images1.append(Images.pixel_array)
```

`Images1`是仅包含图像的列表。我们现在可以存储图像或“腌制它”。

前一种技术适合于将它提供给网络，但不幸的是，如果有人在每个图像上都有标签，这可能不是很有效，而且据我所知，dcm 格式还不会被标记！因为 dcm 格式考虑了收集医学图像的初始阶段，还没有开始被专家(例如，医生、放射科医师)标记。该方法保持图像大小不变，但这是确保像素数据不变的好方法。

另一方面，使用名为 **XnView** 的软件来查看、组织和转换(。dcm)格式。对于只需要查看图像的人来说，使用该软件要容易得多。出于某种原因，有人可能需要将图像转换成 PNG 格式。将图像从 DICOM 格式转换成 PNG 格式时，我使用了。dcm 图像和实现的大小(。dcm)是 3.3MB，而(.png)只有 630KB，这是一个很大的压缩。XnView 有一个功能，用户可以通过批处理导出图像，这使得将批处理转换为 PNG 变得很容易，然后只使用 PNG 图像，而不使用以前使用 python 的方法。

最后，数据集(在我们的情况下是 DICOM 图像)考虑了建立可以分类的健壮模型的主要重要部分之一。因此，当我们处理 DICOM 格式的图像时，在不丢失图像或操纵图像特征的情况下正确提取图像是非常重要的，尤其是在处理医学图像时。