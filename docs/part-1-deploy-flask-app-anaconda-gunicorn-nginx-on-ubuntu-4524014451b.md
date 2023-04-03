# 第 1 部分:在 Ubuntu 上部署 Flask App+Anaconda+guni corn+Nginx

> 原文：<https://medium.com/analytics-vidhya/part-1-deploy-flask-app-anaconda-gunicorn-nginx-on-ubuntu-4524014451b?source=collection_archive---------6----------------------->

![](img/1a11caa65a6dd2114a00ea29438c7c89.png)

萨法尔·萨法罗夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

*一个非常热情的机器学习开发者创造了一个非常有趣的 ML 模型。由于无法控制与他人分享的渴望，他决定创建 API 并与他人分享。经过不太长时间的寻找，他发现了一个巨大的框架——烧瓶。API 是在很短的时间内创建的，并通过运行测试。py 文件照常。开发人员非常高兴，他将它部署在服务器上。现在，他的许多同事开始抨击 API，只是为了检查他为什么对此如此大惊小怪。迎接他们的是惊喜——“服务器出错”。*

Flask 提供的内部服务器用于本地测试，而不是生产。我们需要创建客户机-服务器框架来处理用户请求。

Flask:用于 web 应用的 python 框架

Gunicorn:处理客户机请求的应用服务器

Nginx:网络服务器

在本文中，我们不会深入讨论 flask a 或 gunicorn 或 nginx 的细节。我们将看到如何部署和运行 flask 应用程序。请注意，应用程序和 web 服务器还有其他选项。

1.  **蟒蛇安装**

首先，更新 ubuntu 包

```
sudo apt update 
sudo apt-get upgrade
sudo apt-get install build-essential
```

安装 Anaconda/Miniconda

```
cd ~
wget [https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh)
bash Anaconda3-2019.10-Linux-x86_64.sh
```

创建 conda 环境并安装所需的软件包

```
conda create -n myapp python=3.7.3
conda activate myappconda install flask
conda install pandas
'
'install as per requirements
'
```

**2。Gunicorn 安装**

```
pip install gunicorn
```

**3。应用创建:api_app.py**

```
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello World"

if __name__ == "__main__":
    app.run(host='0.0.0.0')
```

要测试应用程序:

```
python api_app.py
```

应用程序将在 machine_ip:5000 上运行，在上述情况下，请转到 url -http://localhost:5000

**4。应用程序和 Gunicorn 之间的连接**

创建连接器文件以映射 flask 应用程序和 gunicorn 服务器。

```
vi connector.pyfrom api_app import app
if __name__ == "__main__":
      app.run()
```

现在通过提供连接器文件和应用程序模块名来运行 gunicorn 服务器

```
gunicorn --bind 0.0.0.0:5000 connector:app
```

现在，应用程序将运行在相同的地址:http://localhost:5000 上，但是客户端请求将由 app server -Gunicorn 处理。

在下一篇[文章](/@sarang0909.bds/part-2-deploy-flask-app-anaconda-gunicorn-nginx-on-ubuntu-b12fc4199c59)中，我们将看到如何配置 web 服务器-nginx 以实现负载平衡等。

如果你喜欢这篇文章或有任何建议/意见，请在下面分享！

[领英](https://www.linkedin.com/in/sarang-mete-6797065a/)