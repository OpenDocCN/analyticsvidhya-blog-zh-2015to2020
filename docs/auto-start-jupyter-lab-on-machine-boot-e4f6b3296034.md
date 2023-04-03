# 机器启动时自动启动 Jupyter Lab

> 原文：<https://medium.com/analytics-vidhya/auto-start-jupyter-lab-on-machine-boot-e4f6b3296034?source=collection_archive---------3----------------------->

![](img/471ae71b9a3ba0c6c28bfb450b29bbae.png)

Jupyter 实验室服务

在[之前的文章](/analytics-vidhya/setting-up-jupyter-lab-instance-on-google-cloud-platform-3a7acaa732b7)中，我们已经看到了如何在 Google 云平台的虚拟机上安装 jupyter lab。在本文中，我们将了解如何启用 jupyter lab 作为服务，这样用户就不必手动运行命令来启动 jupyter lab 服务。

通过 SSH 登录虚拟机，并执行以下命令:

```
sudo mkdir -p /opt/jupyterlab/etc/systemd
sudo touch /opt/jupyterlab/etc/systemd/jupyterlab.service
```

打开文件“/opt/jupyterlab/etc/systemd/jupyterlab . service”并添加以下内容

```
[Unit]
Description=JupyterLab
After=syslog.target network.target[Service]
User=root
**Environment="PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games:/opt/puppetlabs/bin"
ExecStart=/usr/local/bin/jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root**[Install]
WantedBy=multi-user.target
```

请验证粗体文本，并确保根据您的环境将其替换为正确的路径。

保存文件并执行以下命令。

```
sudo ln -s /opt/jupyterlab/etc/systemd/jupyterlab.service /etc/systemd/system/jupyterlab.service
sudo systemctl daemon-reload
sudo systemctl enable jupyterlab.service
sudo systemctl start jupyterlab.service
sudo systemctl status jupyterlab.service
```

当您运行最后一个命令时，您将看到一个令牌。

访问网址“http:// < <ip>>:8888”并设置从上述步骤中检索到的令牌，并提供您的密码，这样我们就不需要每次都使用令牌。</ip>

现在，停止虚拟机并再次启动它。一旦虚拟机启动，请访问“http:// < <ip>>:8888 ”,它将询问密码以继续。</ip>