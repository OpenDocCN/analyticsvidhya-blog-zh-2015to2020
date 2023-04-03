# Raspberry Pi 4 中推理 Azure 自定义视觉对象检测

> 原文：<https://medium.com/analytics-vidhya/inferencing-custom-vision-in-raspberry-pi-4-2088cbc299c?source=collection_archive---------27----------------------->

![](img/473ec12aa81e78d6763629d1b844cf7e.png)

# 使用 raspberry pi 4 在没有 docker 容器的情况下推断工作场所安全模型

当从 custom vision 导出模型时，可以选择使用 tensorflow 构建模型，并将其嵌入 docker 以用于各种操作系统，如 Raspberry PI、linux 和 windows。为此，我们需要使用 tensorflow 下载选项或其他移动选项来重新训练模型。

要开始使用 raspberry pi 4，首先通过安装操作系统让 Raspberry PI 4 进入工作状态。接下来是更新 python3，请点击下面的链接[https://github . com/insta bot-py/insta bot . py/wiki/Installing-Python-3.7-on-Raspberry-Pi](https://github.com/instabot-py/instabot.py/wiki/Installing-Python-3.7-on-Raspberry-Pi)

安装 python3 后，请按照说明安装 tensorflow。安装 tensorflow 之前，请更新操作系统。sudo apt-get 升级

# 张量流装置

在 raspberry pi 4 中打开一个终端，键入以下命令

```
sudo apt-get install libatlas-base-dev 
sudo pip3 install tensorflow
```

测试 tensoflow 是否已安装。在 Raspberry pi 桌面的主菜单中进入编程，导航到 Thonny python ide 并选择它。单击“新建”并键入以下内容:将 tensorflow 导入为 tf print(tf。**版本**

现在单击 Run，tensorflow 的版本应该显示在底部的 Shell 输出窗口中。

更多详情请点击此链接:[https://magpi . raspberry pi . org/articles/tensor flow-ai-raspberry-pi](https://magpi.raspberrypi.org/articles/tensorflow-ai-raspberry-pi)

现在，Raspberry Pi 已经准备好编写推理代码了。

转到链接:[https://github . com/balakreshnan/work place safety/tree/master/TensorFlowModelsforotherplatform](https://github.com/balakreshnan/WorkplaceSafety/tree/master/TensorFlowModelsforotherplatform)下载 ARM 的模型。[https://github . com/balakreshnan/work place safety/blob/master/TensorFlowModelsforotherplatform/34 f 05 b 15 e 0 be 4 bcfabbcc 0a 216183 EC 8。DockerFile.ARM.zip](https://github.com/balakreshnan/WorkplaceSafety/blob/master/TensorFlowModelsforotherplatform/34f05b15e0be4bcfabbcc0a216183ec8.DockerFile.ARM.zip)

将文件下载到/home/pi/examples 文件夹中。将文件解压缩到该文件夹中。现在创建一个名为 ws (workplacesafety)的新文件夹。现在将所有解压后的内容复制到 ws 文件夹中。下载一些背心图片来测试推理模型。我从 google.com 下载的

现在是时候创建推理文件了。创建名为 detect.py 的新 python 文件

```
import json 
import os 
import io 
# Imports for the REST API 
from flask import flask, request, jsonify 
# Imports for image processing 
from PIL import Image 
# imports for prediction 
from predict import initialize, predict_image, predict_url if __name__ == '__main__' 
    # load and initialize the mode 
    initialize() 
    image = Image.open("vest1.jpg") 
    # change the file name to what you downloaded. 
    result = predict_image(image) 
    Print() 
    i = 0 
    for tag in result['predictions']: 
        print("ID: " + str(tag['tagId']) + " Object Detected: " + tag['tagName'] + " Confidence: " + '{0:f}.format(tag['probability'])) 
        print()
```

保存文件，运行并输出显示检测到的目标及其概率。

*最初发表于*[*【https://github.com】*](https://github.com/balakreshnan/WorkplaceSafety/blob/master/InferencinginRasperryPi4.md)*。*