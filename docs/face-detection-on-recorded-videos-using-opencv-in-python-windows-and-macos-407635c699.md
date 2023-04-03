# 在 Python-Windows 和 macOS 中使用 OpenCV 对录制的视频进行人脸检测

> 原文：<https://medium.com/analytics-vidhya/face-detection-on-recorded-videos-using-opencv-in-python-windows-and-macos-407635c699?source=collection_archive---------6----------------------->

![](img/e3d1107f92158d679db39f0f5a6e2225.png)

视频来源— [链接此处](https://www.pexels.com/video/people-having-a-meeting-and-discussion-at-work-3248990/)

人脸检测是一种计算机技术，它利用人工智能的力量来定位图像或视频中人脸的存在。随着开源项目的发展，现在可以识别人脸，而不考虑肤色、肤色、面部位置和动作。

我们将学习如何构建一个人脸检测系统，该系统将预先录制的视频作为输入，并识别其中的人脸。这个项目类似于实时人脸检测项目([链接此处](/analytics-vidhya/real-time-webcam-face-detection-system-using-opencv-in-python-windows-and-macos-86c31fddd2bc))——不同的是后者以直播的网络视频流作为输入。你可能会发现内容有些重叠。

# 要安装在 pip 中的库(Windows + macOS)

1.  **OpenCV(开源计算机视觉)库:**它的建立是为了帮助开发者执行与计算机视觉相关的任务。

```
install opencv-python
```

接下来，我们需要安装人脸识别 API

**2。dlib 库:** dlib 是通过预先训练好的模型建立起来的，用来定位面部标志点。

```
pip install dlib
```

![](img/30943cace8cbcaf2d3506908db3fd8cf.png)

**3。face_recognition 库:** face_recognition 是一个开源项目，也被称为人脸识别最直观的 API。

```
pip install face_recognition
```

# 让我们来看看 Python 中的一些动作

现在，我们已经安装了所需的库。我们将开始实现如下所示的代码。我解释了每个代码块，以帮助您理解后台发生了什么。

请随意跳到这一页的末尾，获得完整代码的链接。

**步骤 1:** 打开 Spyder

**步骤 2:** 导入库

```
import cv2
import face_recognition
```

**第三步:**参考保存在硬盘上的视频文件(mp4 格式)

```
cap= cv2.VideoCapture(**<enter file path.mp4>**)
```

**第四步:**初始化需要的变量。这些变量将在稍后的代码中填充

```
face_locations = []
```

**第五步:**其工作方式是将视频分成帧，代码一次读取一帧。在每一帧中，我们使用上面导入的 API 来检测人脸的位置。对于每个检测到的人脸，我们定位坐标，并在它周围画一个矩形，然后将视频发布给观众。

完整的代码如下所示——代码下面有解释

```
while True:
    # Grab a single frame of video
    ret, frame = cap.read() # Convert the image from BGR color (which OpenCV uses) to RGB   
    # color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1] # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame) for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0,  
        255), 2) # Display the resulting image
    cv2.imshow('Video', frame)

    # Wait for Enter key to stop
    if cv2.waitKey(25) == 13:
        break
```

第一区:

```
# Grab a single frame of video
 ret, frame = cap.read()# Convert the image from BGR color (which OpenCV uses) to RGB color #(which face_recognition uses)
 rgb_frame = frame[:, :, ::-1]
```

这里，我们一次处理一帧。使用 cv2 库提取帧，cv2 库以 BGR(蓝-绿-红)颜色捕获帧，而面部识别库使用 RGB(红-绿-蓝)格式。因此，我们翻转框架的颜色代码。

**模块 2:**

```
face_locations = face_recognition.face_locations(rgb_frame)
```

这里，我们定位帧中存在的面部的坐标。列表 face_locations 由检测到的面的 x、y 坐标以及宽度和高度填充。

**第三块:**

```
for top, right, bottom, left in face_locations:
 # Draw a box around the face
 cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
```

这里，我们在每个捕捉到的人脸周围画一个矩形。该矩形从 x 和 y 坐标(在本例中为左侧和顶部)开始，延伸到检测到的面部的宽度和高度(在本例中为右侧和底部)。

**第四街区:**

```
cv2.imshow(‘Video’, frame)if cv2.waitKey(25) == 13:
        break
```

结果图像(帧)被释放给查看者，并且循环继续运行，直到用户按下键盘上的回车键。

**第六步:**所有拍摄的视频必须发布。

```
video_capture.release()
cv2.destroyAllWindows()
```

# 在命令行中运行程序

下一步是将文件保存在。py 格式并在命令行/Anaconda 提示符下运行它。

我在 Anaconda 提示符下运行它，首先使用命令 cd 导航到该文件夹。

我在 Anaconda 提示符下运行它，首先使用命令 cd 导航到该文件夹。

```
cd <folder path>
```

运行 python 文件

```
python filename.py
```

您将看到一个弹出窗口，视频正在播放。视频可能会很慢，这是因为 OpenCV 中的帧数通常很大。但是，如果您[将视频保存](/@venkatesh.chandra_75550/saving-output-of-object-recognition-in-macos-opencv-python-5914bb5d9ca8)在硬盘上，那么写入的视频速度并不慢，并且与输入视频的 fps(每秒帧数)相匹配。

![](img/0a52c86b6935aae9cf5b67e7ef5b0831.png)

你可以在这里下载更多免版税视频[。](https://www.pexels.com/)

**没错！您已经成功编写了一段代码，可以在视频中检测人脸。**

**打开 Snapchat 过滤器，尝试在视频上进行实验，并在下面的评论区告诉我效果如何。**

如果你有任何问题，请在评论区告诉我。

# 编码

[](https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/tree/master) [## chandravenky/Python 中的计算机视觉对象检测

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/chandravenky/Computer-Vision---Object-Detection-in-Python/tree/master) 

# 相关链接

[**Python 中的实时人脸检测系统——Windows 和 macOS**](/@venkatesh.chandra_75550/real-time-webcam-face-detection-system-using-opencv-in-python-windows-and-macos-86c31fddd2bc)

[**使用 OpenCV 的 Python 中的车辆检测— Windows 和 macOS**](/@venkatesh.chandra_75550/vehicle-car-detection-in-real-time-and-recorded-videos-in-python-windows-and-macos-c5548b243b18)

[**Python 中的行人检测使用 OpenCV — Windows 和 macOS**](/@venkatesh.chandra_75550/person-pedestrian-detection-in-real-time-and-recorded-videos-in-python-windows-and-macos-4c81142f5f59)

[**在 macOS 中保存物体识别输出**](/@venkatesh.chandra_75550/saving-output-of-object-recognition-in-macos-opencv-python-5914bb5d9ca8)

# 去哪里找我🤓

1.  在[LinkedIn](https://www.linkedin.com/in/venkateshchandra/)/[GitHub](https://github.com/chandravenky)/[我的网站](http://chatraja.com/)上与我联系
2.  感觉大方？在这里给我买杯[咖啡](https://www.buymeacoffee.com/chandravenky) ☕️