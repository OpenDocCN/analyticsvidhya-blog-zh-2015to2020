# 从本地 IDE 在云中开发

> 原文：<https://medium.com/analytics-vidhya/developing-on-a-virtual-machine-from-your-local-ide-7b26b295cccf?source=collection_archive---------2----------------------->

## 从本地 IDE 中设置在虚拟机上运行的实时开发环境的演练

![](img/eb51202456e449bb5cc55e0e06d29e01.png)

照片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [israel palacio](https://unsplash.com/@othentikisra?utm_source=medium&utm_medium=referral) 拍摄

# 介绍

任何对大量数据建模或进行某种深度学习工作感兴趣的人都会很快达到他们本地机器的极限。一旦你写了一些代码，在一个虚拟机(VM)上执行它以获得一次性结果或者预定的[生产目的](https://blog.cloudera.com/putting-machine-learning-models-into-production/)就相对容易了。然而，在 VM 中进行远程开发和现场代码更改实验需要几个步骤。一个简单的解决方案是通过连接到远程机器的 Jupyter 或 Google Colab 笔记本在您的浏览器中工作。笔记本是快速一次性分析的好工具，但是[在软件工程最佳实践方面没有提供太多](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/)。更具体地说，健壮且可重复的机器学习产品或实验[需要比笔记本](https://www.thoughtworks.com/insights/blog/coding-habits-data-scientists)更好的基础来编写生产级代码。

这导致了一个简单的愿望，即希望在选择的集成开发环境(IDE)中工作，同时使用虚拟机的计算能力和资源。我很快发现 RStudio 和 PyCham Professional 都有远程开发的选项，并发现这有多难？然而，我无法找到所需的一点一滴的清晰的端到端描述。我决定为未来的自己记录这些步骤，并希望其他人会觉得有用。

在这个例子中，我将使用我最喜欢的黑客编辑器 [Atom](https://atom.io) 进行 Python 开发，该编辑器由运行在[谷歌云平台](https://cloud.google.com/gcp/) (GCP)上的 [Hydrogen](https://atom.io/packages/hydrogen) 驱动。对于替代设置，下面的步骤应该有许多相似之处。如果你选择的武器是带有 [RStudio](https://cloud.google.com/solutions/running-rstudio-server-on-a-cloud-dataproc-cluster) 、 [PyCharm](https://www.jetbrains.com/devops/google-cloud/) (或另一款 JetBrains 产品)的 GCP，你的生活会轻松一点。

# 设置虚拟机

为了与虚拟机通信，我们需要某种形式的认证。这可以用任何 SSH 工具来完成。在 Mac 上，我们可以通过终端用包含的`ssh-keygen`创建一个 SSH 密钥对:

```
$ cd ~/.ssh/
$ ssh-keygen -m PEM -t rsa -C "GCP_username"
```

确保把它放在一个合理的地方(比如`~/.ssh`)，给它一个文件名，然后想一个密码短语。您可以使用以下命令查看生成的公共 SSH 密钥:

```
$ cat ~/.ssh/filename.pub
```

使用 bash 脚本或终端命令启动虚拟机，并确保机器上安装了 Jupyter:

```
$ pip install jupyter
```

接下来，确保虚拟机接受身份验证。在您的本地计算机上，打开浏览器并转到 GCP 控制台。导航到您的虚拟机:计算引擎→虚拟机实例→元数据→ SSH 密钥。单击 edit 并在字段中粘贴您的公共 SSH 密钥，保存它。

要通过外部 SSH 连接访问虚拟机，我们需要[为其分配一个外部 IP 地址](https://cloud.google.com/compute/docs/ip-addresses/reserve-static-external-ip-address)。单击虚拟机名称→编辑→网络接口→外部 IP。不要忘记点击页面底部的保存按钮。

我们可以通过从本地终端登录虚拟机来检查连接和身份验证是否正常:

```
[$](https://gist.github.com/b7381149624599ea23f70d93637158b2) ssh -i ~/.ssh/place_of_ssh_keys gcp_username@external_ip_address
```

# 创建 Jupyter 内核和 SSH 隧道

当在终端中连接到虚拟机时，在您选择的端口上启动 Jupyter 内核，我使用 8888。此外，没有监视器意味着我们不需要在虚拟机中启动浏览器；)

```
$ jupyter-notebook --no-browser --port=8888
```

为了与虚拟机上运行的 Jupyter 内核通信，我们使用从本地机器到虚拟机的端口转发。这可以通过创建一个 [SSH 隧道](https://www.ssh.com/ssh/tunneling/example)来实现。在新的本地终端窗口中写入:

```
ssh -i ~/.ssh/filename -N -L localhost:8888:localhost:8888 gcp_username@external_ip_address
```

在我的例子中，没有消息出现在终端中，这可能有点混乱，只要保持终端窗口打开。

# 从 IDE 连接到内核

打开 Atom，安装氢气包。转到首选项→包装→氢气→设置，并将以下内容添加到`Kernel Gateways`字段:

```
[{
  "name": "Jupyter Remote Kernel",
  "options": {"baseUrl": "http://localhost:8888"}
}]
```

在你的代码文件中打开 Atom 命令面板`(cmd+shift+p)` → `Hydrogen: Connect to Remote Kernel`。您的 Jupyter 远程内核应该会出现。在虚拟机中启动 Jupyter 内核时，您可以使用终端中显示的令牌进行连接。如果您在多文件项目中工作，您可以简单地用`Hydrogen: Connect to Existing Kernel`将这些文件连接到内核。你可以启动[各种 Jupyter 内核](https://github.com/jupyter/jupyter/wiki/Jupyter-kernels)在你最喜欢的语言中工作，比如 R 或 JavaScript。

为了在本地机器和虚拟机之间同步、编辑或上传文件，我们需要一些权限。在连接到虚拟机的终端中，转到用户帐户中的代码目录，将所有者从 root 更改为登录用户:

```
$ sudo chown -R gcp_username:gcp_username path/to/code/folder
```

您可以使用以下命令检查当前目录下所有文件夹和文件的所有权和权限:

```
ls -l
```

# 设置文件同步

为了在本地开发并在虚拟机上运行代码，我们需要实时反映代码的变化。这可以通过在本地机器和 VM 之间设置 SSH 文件同步来实现。我使用 Atom 包`Remote FTP`，它非常好用，但是你也可以使用你选择的任何其他 SSH 包或软件。

配置软件包进行同步，我在我的`.ftpconfig`文件中使用以下设置:

```
{
    "protocol": "sftp",
    "host": "external_ip_address",
    "port": 22,
    "user": "gcp_username",
    "pass": "",
    "promptForPass": false,
    "remote": "path/to/remote/code/folder/on/vm",
    "local": "path/to/local/code/folder",
    "agent": "",
    "privatekey": "~/.ssh/filename",
    "passphrase": "your_passphrase",
    "hosthash": "",
    "ignorehost": true,
    "connTimeout": 10000,
    "keepalive": 10000,
    "keyboardInteractive": false,
    "keyboardInteractiveForPass": false,
    "remoteCommand": "",
    "remoteShell": "",
    "watch": [],
    "watchTimeout": 500
}
```

您可以在保存时启用文件同步，这将实时反映您的代码从本地到虚拟机的所有更改，以便进行实时开发。

就这样，你可以走了！

# 多方面的

如果您想更频繁地这样做，您可以让自己的生活更轻松，并在您的`~/.bashrc`文件中添加一些别名来加速终端命令。例如:

```
alias gcp_ssh = 'ssh -i ~/.ssh/filename gcp_username@external_ip_address'alias port_forward = 'ssh -i ~/.ssh/filename -N -L localhost:8888:localhost:8888 gcp_username@external_ip_address'alias give_me_the_power = 'sudo chown -R gcp_username:gcp_username path/to/code/folder'
```

重新启动虚拟机后，如果出现与已验证主机相关的验证错误，您可以通过删除`known_hosts`文件并再次登录到机器来重新生成该文件:

```
$ cd ~/.ssh/
$ rm -f known_hosts
$ ssh -i ~/.ssh/filename gcp_username@external_ip_address
```

如果您想将 Python 项目作为一个包运行或导入，您可以通过虚拟机上的 pip 安装本地代码:

```
$ cd path/to/code/folder
$ python3 -m pip install -e .
```

您可以通过浏览器中的`localhost:8888`并使用您的令牌或密码登录，通过本地机器上的 Jupyter 内核浏览虚拟机上的文件夹结构。您可以使用`gcloud`命令或任何 SSH 客户端将文件夹和文件从虚拟机下载回您的本地机器。

您也可以从 Jupyter 浏览器下载到本地。如果你想下载完整的文件夹，而不仅仅是文件，你可以启动一个终端窗口。新建→终端:

```
$ zip -r path/to/folder name_of_zipfile
```

这将为您的文件夹创建一个 zip 文件，您可以像其他文件一样通过笔记本用户界面选择和下载该文件。

如果你遇到任何问题，请随时给我留言！