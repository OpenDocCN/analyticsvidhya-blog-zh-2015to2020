# Docker —开发、调试、部署

> 原文：<https://medium.com/analytics-vidhya/docker-dev-debug-deploy-15b232eed53e?source=collection_archive---------10----------------------->

![](img/1928502cb67801c8480bce3ae53db78f.png)

照片由[凯特·坦迪](https://unsplash.com/@katetandy?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 为什么

作为一名从事几个项目的开发人员，我想找到一种方法-

*   我的安装时间会尽可能快
*   与团队成员分享我的代码和项目设置会很容易
*   我希望我的开发环境尽可能接近生产环境

所以我开始寻找这样做的方法，首先我想到了`Git`，但是`Git`的问题是设置时间不是最优的-
我如何自动配置`Environment Variables`？或者为我的项目同步`Git hooks`，如果我的项目正在使用新的编程语言或版本呢？我不想在本地电脑上开始安装和管理版本..

也不想为任何解决方案买单。

然后我开始考虑使用`Docker`作为我的开发环境，我可以将`Docker`作为“个人开发服务器”运行，并在`Docker Image`中配置我需要的一切——编译器/解释器、环境变量、依赖项等等，当我完成编码时，我可以在我的生产环境中使用相同的`Docker image`

# 什么是`Docker`

"`Docker`是一个用于开发、发布和运行应用程序的开放平台。`Docker`使您能够将应用与基础设施分离，从而快速交付软件。借助`Docker`，您可以像管理应用程序一样管理基础设施。通过利用`Docker's`方法快速交付、测试和部署代码，您可以显著减少编写代码和在生产中运行代码之间的延迟。”
来自`Docker`官方文档，你可以在这里阅读更多

# 建设

我用这个`Dockerfile`作为[这里](https://docs.docker.com/engine/examples/running_ssh_service/)的模板，为了构建我的`Docker Image`，我安装了`openssh-server`，这样我就可以像开发服务器一样使用它，并与我的队友分享。

```
FROM python:3.6
ARG GIT

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:MYSUPERSECRETROOTPASSWORD' | chpasswd

# According to your linux distribution this line my differ
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

RUN echo ' \n\
git clone $GIT ~/code \n\
cd ~/code \n\
git pull \n\
pip install -r requirements.txt' > ./run.sh

# Also configure hooks, environment variables and more

RUN sh ./run.sh

RUN echo '/usr/sbin/sshd -D' >> ./run.sh

CMD ["sh", "./run.sh"]
```

当执行`docker build`时，它将`git clone`我的项目并安装`pip`依赖项以及我需要做的任何事情，我还定义了相同的`./run.sh`文件，每次我使用`docker run`时都要执行，以便让我的 docker 保持最新的新提交等等

# 奔跑

我使用这个命令来运行我的映像

```
> docker run -dP my_cool_docker
37d6e53cb27396467a10c7361d319d28d0197a7b5dc7347bb39c251dff7403dc

> docker port 3
22/tcp -> 0.0.0.0:32768

> ssh root@localhost -p 3276
The authenticity of host '[localhost]:32768 ([::1]:32768)' can't be established.
ECDSA key fingerprint is SHA256:z4x3yWVSJZAoswgEa0utt5jSv0Mt0Ex6sMY8a4CFCnE.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added '[localhost]:32768' (ECDSA) to the list of known hosts.
root@localhost's password:
Linux 37d6e53cb273 4.9.184-linuxkit #1 SMP Tue Jul 2 22:58:16 UTC 2019 x86_64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
root@37d6e53cb273:~# ls -l
total 4
drwxr-xr-x 1 root root 4096 Dec 14 09:45 code
```

如您所见，我使用`detach`标志和`expose all`端口运行`image`，然后使用`ssh`通过暴露的端口本地主机，输入您的超级秘密 root 密码，我们有了一个个人开发服务器，它实际上是运行在您个人 PC 上的`docker container`！

# 优势

*   一次设置满足我的所有需求
*   相同的开发和生产环境
*   易于与队友分享

# 不足之处

*   如果您的`container`崩溃，您可能会丢失数据
*   适合那些在生产中使用容器的人

享受😋

> 喜欢这个帖子？你想访问我的私人 github 项目？
> 通过 [*补丁*](https://www.patreon.com/eranelbaz) 支持我

*原载于 2019 年 12 月 21 日*[*https://dev . to*](https://dev.to/eranelbaz/docker-dev-debug-deploy-199l)*。*