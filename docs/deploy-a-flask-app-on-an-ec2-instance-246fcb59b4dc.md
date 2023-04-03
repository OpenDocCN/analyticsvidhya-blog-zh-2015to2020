# 在 EC2 实例上部署 Flask 应用程序

> 原文：<https://medium.com/analytics-vidhya/deploy-a-flask-app-on-an-ec2-instance-246fcb59b4dc?source=collection_archive---------15----------------------->

![](img/7de3970311d6e6e5dcafd3376967bc20.png)

# 设置 EC2 实例

1.  从 AWS 仪表板转到 EC2 并启动一个实例。
2.  选择您喜欢的亚马逊机器映像(AMI)。我的首选 AMI 是 Ubuntu 服务器。
3.  继续选择一个实例类型。对于简单的项目，我通常使用 t2.micro，所以根据您的需求选择一个实例类型。
4.  继续到“配置安全组”,不做任何修改。安全组是最重要的部分，因为如果没有正确配置传入和传出流量，您可能无法访问您的 web 应用程序。在类型中选择“所有流量”，并允许来源为“任何地方”。
5.  接下来，继续完成实例的启动。对于良好的安全措施，您希望有更好的入站和出站安全组规则，但由于本指南是出于教程的目的，所以我对安全规则很在行。
6.  在完成之前选择您的密钥对(如果您还没有密钥对，您可以选择创建一个)。

祝贺您，您已经成功创建并启动了 EC2 实例！

通过单击它并选择顶部的“Connect ”,连接到正在运行的实例。将 ssh 命令复制到“Example”下，并粘贴到您的终端中。除非您位于与相同的目录中，否则 SSH 命令将不起作用。pem 文件。

现在，您应该通过命令行/终端连接到您的实例。

# 准备托管应用程序的实例

最重要的是。我们需要安装 git，python 和 pip 来继续我们的项目。

## 从更新包索引开始:

```
sudo apt update
```

## 安装 git:

```
sudo apt install git
```

## 安装 Python 3.8:

```
sudo apt install python3.8
```

## 检查安装的 Python 版本:

```
python --version
```

## 安装 pip3:

```
sudo apt-get install python3-pip3
```

## 在虚拟机/EC2 实例中克隆 Flask 应用程序的 GitHub repo:

```
git clone [your repo link]
```

## cd 到您的新目录:

```
cd [directory name]
```

## 安装 requirements.txt 文件:

```
pip3 install -r requirements.txt
```

更多关于 requirements.txt 文件[这里](/@boscacci/why-and-how-to-make-a-requirements-txt-f329c685181e)

(只需在您的。txt 文件，如果您在安装依赖项时经常遇到错误)

如果您没有创建 python 项目，也不用担心。使用 vim 创建一个简单的 python 文件:

```
mkdir projcd projtouch app.pyvi app.py
```

进入 Vim 后，点击“I”进入“插入模式”,然后粘贴以下内容:

```
from flask import Flaskapp = Flask(__name__)@app.route('/')
def index():
    return "The App is running!"if __name__ == '__main__':
    app.run(debug=True)
```

按 esc 键击":"然后按" x "和" enter "键。

## 我们将在虚拟环境中运行 Flask 应用程序，因此让我们安装 virtualenv:

```
sudo pip3 install virtualenvwrappersudo su appsvi .bashrc
```

## 添加以下内容:

```
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_VIRTUALENV_ARGS='--no-site-packages'source /usr/bin/virtualenvwrapper.sh
```

## 创建虚拟环境:

```
mkvirtualenv myapp
```

## 在您的虚拟环境中安装 gunicorn:

```
pip3 install gunicornexit
```

## 接下来，我们需要在虚拟机上安装一个 web 服务器:

```
sudo apt-get install nginxsudo vi /etc/nginx/nginx.conf
```

替换此行:

```
user  nginx;
```

使用:

```
user  apps;
```

在 http 块中添加以下内容:

```
server_names_hash_bucket_size 128;
```

## 为我们的站点定义一个服务器块:

```
sudo vi /etc/nginx/conf.d/virtual.conf`
```

将此粘贴到内部:

```
server {
    listen       80;
    server_name  your_public_dnsname_here;location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

## 启动服务器:

```
sudo /etc/rc.d/init.d/nginx start
```

## 启动 Gunicorn 流程，为我们的 Flask 应用提供服务:

```
sudo su appsgunicorn app:app -b localhost:8000 &
```

应用程序应该启动并运行。

返回到浏览器上的 EC2 实例，复制 EC2 的公共 DNS(IPv4)地址，并将其粘贴到一个新选项卡中。如果成功，浏览器将显示消息“应用程序正在运行！”

*恭喜*您已经在 EC2 实例上成功部署了您的 Flask 应用程序！