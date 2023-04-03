# 使用 OpenCV 的 15 行 python 代码实现人脸检测

> 原文：<https://medium.com/analytics-vidhya/face-detection-with-15-lines-of-python-code-using-opencv-2508e2d2f8f8?source=collection_archive---------13----------------------->

![](img/4424c7448103720db556f78f571cf58d.png)

大家好，欢迎来到令人兴奋的计算机视觉世界，这里的乐趣永无止境。

今天，我们将了解如何在视频中检测人脸。在这里，我们将使用主要的网络摄像头用于学习目的，它肯定可以替换为任何视频文件。

所以，让我们开始行动吧！

## **先决条件:**

**1—OpenCV
2—Haar scade XML 文件**

## **步骤:**

1 —让我们首先打开网络摄像头，阅读视频馈送并在窗口中显示。

2 —添加一个条件，帮助我们退出视频馈送并关闭步骤 1 中使用的显示窗口。

3-检测面部并在面部周围绘制一个矩形。

使用这些步骤，我们就大功告成了。

```
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    cv2.imshow('Video feed', img)
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
```

上面的代码通知导入 cv2 (opencv)进行进一步的处理。在下一行中，我们通过将 0 作为参数来打开主网络摄像头。如果您有一个辅助摄像头，您可以将 0 更改为 1。

下一行有一个无限循环，表示在没有遇到中断条件之前，将视频提要保持在显示窗口中。

在这个循环中，我们读取 VideoCapture 接收到的任何内容，并通过使用 imshow 将其显示在一个名为“Video Feed”的窗口中。

最后，我们添加一个中断条件，就像按下“ESC”键一样，然后中断循环，将它带到最后一行，以释放网络摄像头并破坏显示窗口。

4 —现在是最后一步，让我们来检测人脸。对于这一步，我们将使用下面一行代码加载一个 Haarcascade ' Haarcascade _ frontal face _ default . XML'。

```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

5-为了避免任何干扰，我们将使用下面一行代码将从网络摄像头读取的图像转换为灰度。

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

6-我们将使用下面一行代码使用灰度图像来检测人脸。

```
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
```

faces 变量将具有长度、宽度、高度和宽度。

7-我们将使用这些坐标绘制一个矩形，如下图所示。

8-绘制矩形时传递的参数是实际图像、起点坐标、终点坐标、要使用的颜色、边框宽度。

```
for (x, y, w, h) in faces:
 cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
```

就是这样，我们成功地用 python 写了一个人脸检测程序。恭喜你！！参见下面的完整代码。

```
import cv2face_cascade = cv2.CascadeClassifier(‘haarcascade_frontalface_default.xml’)cap = cv2.VideoCapture(0)while True:
    ret, img = cap.read() gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) faces = face_cascade.detectMultiScale(gray, 1.3, 5) for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2) cv2.imshow(‘image window’, img) k = cv2.waitKey(30) & 0xff if k == 27:
        breakcap.release()
cv2.destroyAllWindows()
```

感谢您访问这篇文章，希望您喜欢。请在下面的评论部分分享您的反馈。直到下一次。

最诚挚的问候