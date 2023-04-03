# 面向所有年龄段儿童的面部识别，第 3 部分

> 原文：<https://medium.com/analytics-vidhya/facial-recognition-for-kids-of-all-ages-part-3-1084be049bf2?source=collection_archive---------19----------------------->

这是一个由 3 部分组成的文章系列的第 3 部分，其中对复杂的计算机视觉和面部识别主题进行了分解，以供初学者遵循。

![](img/dd56db1ae1b73704b4a60db0461ccd0f.png)

*第一条:*[*https://medium . com/@ grokwithrahul/face-recognition-for-kids-of-all-ages-part-1-c 496040 b 2517*](/@grokwithrahul/facial-recognition-for-kids-of-all-ages-part-1-c496040b2517) *第二条:*[*https://medium . com/analytics-vid hya/face-recognition-for-kids-of-all-ages-part-2-3281084091 de【T13*](/analytics-vidhya/facial-recognition-for-kids-of-all-ages-part-2-3281084091de)

现在我们已经理解了 Haar cascades 和 LBPH 模型背后的理论，让我们用代码实现它。

在本文中，我们将使用 Python。首先，您需要下载所需的软件包。打开命令提示符并使用 pip，通过“install”命令下载以下软件包:

> opencv-contrib-python
> 熊猫
> 抱枕

所有的程序和文件都可以方便地下载。向下滚动到“如何使用代码”并从 GitHub 链接分叉/下载。这样，你就可以和我一起解释代码了！

# 面部检测

在本系列的第 1 部分，我们已经讨论了 Viola-Jones(Haar cascade)物体分类方法。选择这种特殊方法的原因包括其准确性、效率和速度。现在，我们将用 Python 实现 haar cascade 方法。*(如果你想了解哈尔级联法是如何工作的，通读本系列的第一篇文章)*

![](img/dd8579d3af80c99fc3efdb5157211d1a.png)

面部检测程序的流程图表示

如果您使用外部网络摄像头，请尝试设置*网络摄像头= cv2。视频捕获(1，cv2。*。这样做将告诉 OpenCV 从您的第二个摄像头读取。

我们从照相机中取出一帧图像，并将其转换成黑白图像。Haar 级联分类器只能对黑白图像进行操作。

然后，我们在黑白图像中检测人脸，并将它们保存在元组‘facesInCurrentFrame’中。数据包括 xCoord、y code、width 和 height。

最后，我们根据 facesInCurrentFrame 中的数据绘制一个矩形。

![](img/162f7c37c6f5b37e3d9abc02cc4a265b.png)

在图像中创建的矩形

# 面部识别

## 照片拍摄

首先，我们需要为 LBPH 模块(在第 2 篇文章中讨论)收集图像以进行训练。我们将在相机的帮助下收集这些图像，并有一个技巧来避免将每张照片裁剪到面部。

在每一帧中，我们检测一张人脸，将帧裁剪到该人脸，并保存裁剪后的图像。这个过程一直持续到 40 个图像被裁剪和保存。

然后，根据 CSV 文件中的数据(转换成熊猫数据帧),我们给每个用户分配下一个可用的 id，然后我们给每个图像加上标题
用户。(id)。(image_number)”。

最后，我们将新用户添加到数据帧中，然后将数据帧导出到 CSV 文件中。所有这些都给了我们 LBPH 模型进行训练的图像。

## LBPH 模型训练

接下来，我们需要用保存的图像训练 LBPH 模块。

在这里，我们只是获取图像，从其名称中提取 id，并将图像及其 ID 提供给 LBPH 模型。

## 认脸！

现在是最精彩的部分。有了训练好的模型，我们就可以开始识别人脸了！

大部分程序与面部检测程序相同。此外，我们将网络摄像头拍摄的当前图像发送到 LBPH 模型，该模型输出一个 ID 号。然后，我们使用 ID 号从 CSV 文件中获取相应的名称。最后，我们显示姓名，以及包围面部的矩形。

太好了！你已经完成了面部识别系列文章！你终于可以实际使用 LBPH 模型和 Haar 级联方法了！

# 如何使用代码

你可以在这里分叉 GitHub repo(你也可以下载为 ZIP 文件):[https://GitHub . com/red tomite/face-recognition-article-series](https://github.com/Redstomite/Facial-recognition-article-series)

对于面部识别，确保首先拍照，其次训练模型，最后运行面部识别程序。要添加更多的人，拍摄新用户的图像，然后训练该模块。用户越多，魔力越大！

如果你正面临着任何问题或发现了程序中的任何漏洞，你可以在这里创建一个新的问题:[https://github . com/red tomite/face-recognition-article-series/issues](https://github.com/Redstomite/Facial-recognition-article-series/issues)。

总之，面部识别是计算机科学中最有趣的领域之一。它的应用范围从数字支付到身份识别系统再到刑事鉴定。最后一点，面部识别正在进入一个非常具有挑战性的时代——一个面具成为新常态的时代。要前进，必须执行创新的解决方案，必须打破界限。