# 基于 Tensorflow 和 OpenCV 的 IP 摄像机视频对象检测

> 原文：<https://medium.com/analytics-vidhya/detecting-objects-on-ip-camera-video-with-tensorflow-and-opencv-e2c25297a75a?source=collection_archive---------2----------------------->

![](img/68dc47efdd4d6f65a464d0f72008b8d2.png)

在我的上一篇文章中，我演示了如何使用 Tensorflow 和 OpenCV 在网络摄像机视频流上检测我的自定义对象。

[](/analytics-vidhya/detecting-custom-objects-on-video-stream-with-tensorflow-and-opencv-34406bd0ec9) [## 利用 Tensorflow 和 OpenCV 检测视频流中的自定义对象

### 在本系列的第一篇文章中，我在 Windows 10 机器上安装了 Tensorflow 对象检测 API，并在…

medium.com](/analytics-vidhya/detecting-custom-objects-on-video-stream-with-tensorflow-and-opencv-34406bd0ec9) 

正如我之前在文章中提到的，我计划为我的花园制作一个鸟类探测器。我发现，对于户外视频，尤其是当要检测的对象在远处时，常规的网络摄像机不适合提供高质量的图片。

看看下面这个 20 秒长的视频，我并排比较了我的网络摄像机和 IP 摄像机的照片。

如果你看了视频，你会发现 IP 摄像头的图像要清晰得多，但是在视频中有一点滞后。网络摄像机的图像更加模糊，但是动作更加生动。此外，网络摄像头的分辨率为高清(1920x1080)，而网络摄像头的分辨率仅为 640x360。然而后者通常为对象检测提供了更好的图像质量。

# IP 摄像头

那么什么是 IP 摄像头呢？它只不过是一台连在网络上的照相机。IP 摄像机将视频信号转换成 IP 数据包，通过网络传输，或者将摄像机上的视频存储在本地存储器上。

![](img/a1189d2ea9b118226c79773b2b0d9c6a.png)

幸运的是，我家里有一台 IP 摄像机，这是我女儿还是婴儿时我们买的，我们在她晚上睡觉时照顾她。这是一款 **OverMax Camspot 3.1** ，相对便宜且老旧的型号，但它将适合体验。

# 用 OpenCV 读取 IP 摄像机视频流

为了开始物体检测，我们必须以某种方式从 IP 摄像机读取视频信号。

幸运的是，OpenCV 以我们打开视频流的通常方式支持开箱即用的 IP 摄像机流:

```
cap = cv2.VideoCapture(˛'rtsp://username:pass[@192](http://twitter.com/192).168.0.137:10554/tcp/av0_1')
```

你只需要提供 IP 摄像头通过网络播放的视频流的 URL、用户名和密码，就可以了。
但是你怎么知道网址呢？
该网址特定于相机制造商。不幸的是，我没有任何关于我的相机的文档，所以我不知道如何找到这个 URL。幸运的是我找到了 iSpyConnect.com，这是一个开源视频监控软件 iSpy 的主页，他们有一个关于 OverMax 摄像机的页面。
在本页中，您将找到多种连接相机的方法。我尝试了几乎所有的方法，最后决定使用两种:一种是基于 rtsp 的*方法，另一种是基于 http 的*方法。

```
# using the rtsp protocol for accessing the video stream
cap = cv2.VideoCapture('rtsp://username:password@192.168.0.137:10554/tcp/av0_1')# using the http protocol for accessing the video stream
cap = cv2.VideoCapture('[http://192.168.0.137:81/videostream.cgi?loginuse=username&loginpas=password&resolution=32'](http://192.168.0.137:81/videostream.cgi?loginuse=username&loginpas=password&resolution=32'))
```

> 请注意，您的用户名和密码是以纯文本形式在网络上广播的。黑客或其他网络攻击者可以很容易地从您的网络包中捕获这些数据。在设计任何系统时，安全性总是一个重要的考虑因素，包括对象检测。**在没有找到通过网络访问相机的安全方法之前，请不要在产品中使用这个例子**。
> 看看基于 SSL 的协议，比如 *rtsps* 或者 *https。安全性超出了我的实验范围。*

# 绕道:RTSP 议定书

在我开始这个实验之前，我对这个方案一无所知。根据维基百科:“*实时流协议(RTSP)* 是一种网络控制协议，设计用于娱乐和通信系统，以控制流媒体服务器”。如果你有兴趣，可以查看下面的维基百科文章。

[](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol) [## 实时流协议

### 实时流协议(RTSP)是一种网络控制协议，设计用于娱乐和…

en.wikipedia.org](https://en.wikipedia.org/wiki/Real_Time_Streaming_Protocol) 

# 估价

我重用了我在上一篇文章中使用的相同对象检测模型和[代码](https://github.com/dschwalm/deeplearning/blob/master/tensorflow/object_detection/webcam_detection_custom.ipynb)来检测 Peppa 和 George。

我只需用更改几行，对象检测工作正常，使用网络摄像头时也是如此。

这太棒了，太简单了！

# 对抗延误

我对结果有一个很大的担忧:有时视频流会有很大的延迟！它并不总是发生，但当它发生时，它会破坏整个对象检测过程。

当延迟发生时，我尝试了不同的协议和访问视频流的方法，但仍然存在巨大的延迟:

*   使用 *rtsp* 协议时，延迟约为 30 秒
*   当使用 *http* 协议时，延迟大约为 7 秒

与我的预期相比，这两个延迟都很大，当需要对视频上发生的事件做出即时反应时，它会阻止任何实时使用。

奇怪的是，当我用厂商提供的应用程序从普通的网络浏览器访问视频时，视频没有任何延迟。
我怀疑 OpenCV 或其访问相机的方式有问题，可能是一些缓冲问题。我尝试了通过谷歌找到的一切，但在这个问题上花了几个小时后，我放弃了。
如果你读到这里，并对如何改善这种延迟有一些想法，请随时 ping 我，或者甚至对我的 github repo 中的代码提出一个拉请求[。](https://github.com/dschwalm/deeplearning/blob/master/tensorflow/object_detection/webcam_detection_custom-ip_camera.ipynb)

感谢你通读我的文章。我发现将对象检测模型从网络摄像机切换到 IP 摄像机非常容易。对于检测户外活动，IP 摄像头似乎是一个更好的选择。我还分享了我在使用 IP 摄像头时遇到的一些延迟的经历，但我无法找出原因，也无法解决。

作为下一步，我将最终解决鸟的探测问题。