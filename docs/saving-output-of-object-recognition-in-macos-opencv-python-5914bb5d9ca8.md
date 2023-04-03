# 在 macOS 中保存对象识别的输出— OpenCV Python

> 原文：<https://medium.com/analytics-vidhya/saving-output-of-object-recognition-in-macos-opencv-python-5914bb5d9ca8?source=collection_archive---------18----------------------->

![](img/4ffd99fa0298d8f009c7d1159a76a2ff.png)

图片来源— [链接](https://www.pexels.com/photo/50mm-camera-lens-canon-laptop-693866/)

这篇文章对那些已经成功完成物体探测任务的人来说很重要。

我对在 Python 中以视频格式在硬盘上保存 OpenCV 上的对象检测输出的过程做了一些研究。我意识到很少有文章解释如何做到这一点的确切方法。在本文中，我们将学习如何在 macOS 系统中保存输出视频。

***提示:在 windows 系统中保存视频是一件痛苦的事情。这个问题与输出编解码器有关，CV 社区经常讨论这个问题。***

让我们以行人检测代码作为参考。

```
import cv2cap = cv2.VideoCapture(**<enter file path.mp4>**)pedestrian_cascade = cv2.CascadeClassifier(cv2.VideoCapture(**<enter file path.xml>**))while True:
    ret, frames = cap.read()
    pedestrians = pedestrian_cascade.detectMultiScale( frames, 1.1, 
    1)for (x,y,w,h) in pedestrians:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frames, 'Person', (x + 6, y - 6), font, 0.5, (0, 
        255, 0), 1)
        cv2.imshow('Pedestrian detection', frames)
    if cv2.waitKey(33) == 13:
        breakcap.release()
cv2.destroyAllWindows()
```

为了记录通过 *imshow* 命令以单个帧显示给用户的内容，我们需要定义输出，其中*循环中的每一帧都覆盖了输出。*

# 将视频写入硬盘的代码— macOS

将输出视频保存在指定路径中的代码如下，粗体部分参照上面显示的代码进行了更改，代码后面有解释

```
import cv2cap = cv2.VideoCapture( **<enter file path.mp4>**)**fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# note the lower case****frame_width = int(cap.get(3))
frame_height = int(cap.get(4))****out = cv2.VideoWriter(<enter file path.mp4>,fourcc , 10, (frame_width,frame_height), True)**pd_cascade = cv2.CascadeClassifier(**<enter file path.xml>**)while True:
    ret, frames = cap.read()pedestrians = pd_cascade.detectMultiScale( frames, 1.1, 1)for (x,y,w,h) in pedestrians:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frames, 'Person', (x + 6, y - 6), font, 0.5, (0, 255, 0), 1)cv2.imshow('Pedestrian detection', frames)
    **out.write(frames)**

    if cv2.waitKey(33) == 13:
        breakcap.release()
**out.release()**
cv2.destroyAllWindows()
```

# 代码的解释

在开始按帧读取视频之前，我们将输出定义如下:

```
**fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# note the lower case****frame_width = int(cap.get(3))
frame_height = int(cap.get(4))****out = cv2.VideoWriter(<enter file path.mp4>,fourcc , 10, (frame_width,frame_height), True)**
```

书面视频的输出需要规范，如文件名、编解码器、每秒帧数(fps)、帧高和帧宽。

**参数说明:**

**fourcc:** 需要声明的编解码器。我们可以这样定义它:

```
**fourcc = cv2.VideoWriter_fourcc('m','p','4','v')# note the lower case**
```

**请注意，你需要用小写字母写编解码器**

另外，**帧宽和帧高**可以参考输入视频/网络摄像头或用户提供。我在代码里用过 FPS 10。但是，您可以使用以下方法获得输入视频的帧尺寸:

```
**frame_width = int(cap.get(3))
frame_height = int(cap.get(4))**
```

**每秒帧数**可以映射到输入视频的 FPS 值，使用:

```
**get(CAP_PROP_FPS) or get(CV_CAP_PROP_FPS)**
```

为了生成彩色视频作为输出，我们使用了 *isColor* 参数。

```
**isColor= True**
```

# 记住

一旦您通过 Anaconda 提示符运行代码，您将看到一个弹出视频，它以帧的形式显示输出。视频以您观看视频的速度在后台写入。

退出输出视频时，视频会保存在指定的路径中，直到输出被流化。

那都是乡亲们！如果你有任何问题，请在评论区告诉我。

# 编码

[](https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/tree/master/macOS%20recording%20codes) [## chandravenky/计算机视觉——Python 中的对象检测

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/tree/master/macOS%20recording%20codes) 

# 相关链接

[**使用 OpenCV 的 Python 中的行人检测— Windows 和 macOS**](/@venkatesh.chandra_75550/person-pedestrian-detection-in-real-time-and-recorded-videos-in-python-windows-and-macos-4c81142f5f59)

[**使用 OpenCV 的 Python 中的车辆检测— Windows 和 macOS**](/@venkatesh.chandra_75550/vehicle-car-detection-in-real-time-and-recorded-videos-in-python-windows-and-macos-c5548b243b18)

[**Python 中的实时人脸检测系统— Windows 和 macOS**](/@venkatesh.chandra_75550/real-time-webcam-face-detection-system-using-opencv-in-python-windows-and-macos-86c31fddd2bc)

[**Python 中录制视频的人脸检测— Windows 和 macOS**](/@venkatesh.chandra_75550/face-detection-on-recorded-videos-using-opencv-in-python-windows-and-macos-407635c699)

# 去哪里找我🤓

1.  在[LinkedIn](https://www.linkedin.com/in/venkateshchandra/)/[GitHub](https://github.com/chandravenky)/[我的网站](http://chatraja.com/)上与我联系
2.  感觉大方？给我买一杯☕️咖啡