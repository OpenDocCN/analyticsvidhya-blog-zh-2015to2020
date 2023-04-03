# OpenCV 装饰器用于简单的图像和视频测试

> 原文：<https://medium.com/analytics-vidhya/opencv-decorator-for-easy-image-and-video-testing-d1677c3dbc3?source=collection_archive---------16----------------------->

![](img/4a527e6039a6a39fdefb360b549aa871.png)

在计算机视觉中，当我们测试我们的图像处理模型，如 VGG-16、Yolo、Res-Net101 或定制模型时，我们面临着使用 cv2.imshow()函数查看图像的重复任务。让我详细说明一下，在计算机视觉中，为了做概念验证，我们首先合成我们的模型。然后对于测试部分，我们将我们的模型放在一个函数中，该函数将一个**单个图像**作为参数，并返回绘制在图像本身上的边界框或某种标识符，以便进行调试。最后，我们可以看到使用 cv2.imshow()函数调试后的图像。

> 如果你是 OpenCV 的新手，可以通过这个链接:[https://OpenCV-python-tutro als . readthedocs . io/en/latest/py _ tutorials/py _ GUI/py _ image _ display/py _ image _ display . html](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)

如果模型(函数)在单个图像上完成的处理工作正常，那么我们前进到在视频流上测试它。这就是装饰者发挥作用的地方。我们可以使用我制作的 decorator 函数，它将向我们的模型函数添加功能，以视频流作为输入，并通过在我们的模型函数上方添加 decorator("@decorator ")行来显示调试后的输出！！

为了更清楚地解释它，我展示了一个简单的函数示例，它在传递彩色图像时返回灰度图像。现在我们想修改我们的函数，这样它将输入一个彩色视频并返回一个黑白视频，同时使用 cv2.imshow()显示结果……我的装饰器马上就会为您完成！！

## 定义基本图像处理功能

图像处理函数 **processFrame** 的例子，它将返回处理后的帧(这里做 BGR 到灰度转换，但我们通常做类似于*在“Res-Net-101”*中传递帧的操作):-

```
def ***processFrame(frame)***:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [frame,gray]
```

## 单幅图像上的测试函数

```
#reading image from disk
inputFrame= cv2.imread('testColorImage.jpg')#processing the input image
img= processFrame(inputFrame)#showing image that we got after processing
cv2.imshow('inputImage',inputFrame)
cv2.imshow('processedImage',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 视频馈送的测试功能

***传统方法**是将我们的函数放在下面的代码中，并在每次测试新函数时复制粘贴其余的代码

```
import numpy as np
import cv2#***videoPath is 0***
cap = cv2.VideoCapture(0)while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() # Our operations on the frame come here
    inputFrame,outputFrame= ***processFrame(frame)*** # Display the resulting frame
    cv2.imshow('inframe',inputFrame)
    cv2.imshow('outframe',outputFrame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```

***装饰者的方法—** 如果我们使用装饰者，这就是我们能做的

```
import numpy as np
import cv2@testFuncOnVideo 
***processFrame(frame)***:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [frame,gray]***processFrame(videoPath=0)***
```

上面带有装饰器的代码将转换这个只占用一帧的函数，现在将处理来自 **videoPath** 源的视频馈送，并显示处理后视频的输出。

我们只需要导入 **testFuncOnVideo** 函数就可以在需要的时候用它作为装饰器。这个函数的代码如下，我刚刚修改了传统的方法，并试图返回一个包装器，它将把函数从处理图像突变为处理视频。

```
def testFuncOnVideo(function):
    *#defining a wrapper function*
    def coverFunction(*args,**kwargs):
        try:
            cap = cv2.VideoCapture(kwargs['videoPath'])
            while(True):
                *# Capture frame-by-frame*
                ret, frame = cap.read()
                if frame is not None:
                    *# Our operations on the frame comes here*
                    output= function(frame)
                    *#output should be a list*
                    if type(output)==type(list()):
                        for i,image in enumerate(output):
                            if image is not None:
                                cv2.imshow('frame'+str(i),image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            *# When everything done, release the capture*
            cap.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass
    return coverFunction
```

因此，就像这个例子一样，我们可以在代码部分找到其他合适的地方，在那里我们可以看到重复并实现一个装饰器。但是应该清楚它实际上做什么。在上面我们使用@ operator 的代码中，该代码相当于:-

```
@testFuncOnVideo 
***processFrame(frame)***:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [frame,gray]||           ||***processFrame(frame)***:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [frame,gray]***processFrame =*** testFuncOnVideo(***processFrame***)
```

> 阅读更多关于装修工的信息:-[https://www.geeksforgeeks.org/decorators-in-python](https://www.geeksforgeeks.org/decorators-in-python/)

# 摘要

*   在编码时，我们会遇到这样一种情况，我们需要添加一些对函数来说并不重要的特性，在这种情况下，装饰者可以完美地工作，而不会影响函数的原始结构
*   我在 OpenCV 中找到了一个甜蜜点，在这里我们可以通过使用@ **testFuncOnVideo，在装饰者的帮助下使我们的调试和测试生活变得更容易。**需要记住的一点是，主机函数应该返回图像列表。