# 连接 Rasa 到网站:一步一步的教程

> 原文：<https://medium.com/analytics-vidhya/connecting-rasa-to-a-website-a-step-by-step-tutorial-e5233268b576?source=collection_archive---------2----------------------->

![](img/728fa6fbcfd8bf14425a601c871a951f.png)

这是写给那些努力将你的 Rasa 机器人连接到网站的人的。

> **如果一开始你没有成功**，**试试**，**再试一次**

第一步:**创建并训练你的机器人**。Rasa 文档是你这一步的朋友。

步骤 2:一旦你的机器人被训练好**，通过运行 Rasa shell 来验证它正在工作**。您可以通过打开 Rasa 文件夹中的终端并运行命令来实现

```
rasa shell
```

第三步:有趣的部分来了。整合部分。为此有许多选择，即您可以使用 Websocket 通道、Rest 通道等。在本教程中，我将使用一个休息通道。

为了在你的网站上显示你的聊天机器人，你需要一个前端和一个连接这个前端和后端的方法，也就是你的 Rasa 机器人。

我使用下面的 GitHub repo 作为我的前端，其余通道用于连接我的前端和后端。我建议你在访问他们之前通读这篇文章。

[](https://github.com/scalableminds/chatroom) [## scalable minds/聊天室

### 基于 React 的 Rasa Stack 聊天室组件。通过创建帐户，为 scalable minds/聊天室的发展做出贡献…

github.com](https://github.com/scalableminds/chatroom) 

之后，将您的 **credentials.yml** 文件更改为

```
rest:
  # pass
```

之后，在你的 rasa 聊天机器人文件夹外为你的前端创建一个新目录。在该目录中，用以下代码创建一个 index.html 作为头和主体部分。

```
<head>
  <link rel="stylesheet" href="https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/dist/Chatroom.css" />
</head>
<body>
  <div class="chat-container"></div>

  <script src="https://npm-scalableminds.s3.eu-central-1.amazonaws.com/@scalableminds/chatroom@master/dist/Chatroom.js"/></script>
  <script type="text/javascript">
    var chatroom = new window.Chatroom({
      host: "http://localhost:5005",
      title: "Chat with Mike",
      container: document.querySelector(".chat-container"),
      welcomeMessage: "Hi, I am Mike. How may I help you?",
      speechRecognition: "en-US",
      voiceLang: "en-US"
    });
    chatroom.openChat();
  </script>
</body>
```

现在所有的编码部分都完成了，下一步是让它们并行工作。这就是端口号的由来。基本上，我们将尝试在本地主机上运行我们的前端和后端。

从 rasa 聊天机器人文件夹中的 **endpoints.yml** 文件中，记下机器人将要运行的端口号。在我的例子中， **endpoints.yml** 文件包含以下代码。

```
action_endpoint:url: "http://localhost:5005/webhook"
```

所以端口号是 5005。

接下来，确保这个端口号与我们上面写的 index.html 文件中提到的端口号相同。

现在进入 rasa 聊天机器人文件夹，在终端中使用下面的命令启动机器人。

```
python -m rasa run --m ./models --endpoints endpoints.yml --port 5005 -vv --enable-api
```

接下来，打开保存 index.html 文件的另一个终端，输入以下代码，在端口 8000 上启动一个本地服务器。

```
python -m http.server 8000
```

现在最后的任务是打开你的浏览器(我推荐 Mozilla Firefox ),在地址栏输入下面的代码。

```
localhost:8000/index
```

找到了。！！希望你让机器人工作了。如果您需要任何帮助，请发表评论。

注意:请务必检查您的浏览器页面，确认没有错误。您可以通过右键单击您的页面并选择 inspect page，然后导航到控制台部分，如果有任何错误，它将显示在那里。

非常感谢大家阅读、喜欢和评论我的文章。请考虑跟随我。干杯😊

👉🏼查看我的网站，[](https://www.milindsoorya.com/)**了解更多更新并保持联系。**

# **你可能也喜欢:-**

*   **[python 中蘑菇数据集的分析与分类](https://milindsoorya.com/blog/mushroom-dataset-analysis-and-classification-python)**
*   **[如何在 Ubuntu 20.04 上用 Python 3 设置 Jupyter 笔记本](https://milindsoorya.com/blog/how-to-Set-up-jupyter-notebook-with-python-3-on-ubuntu-20.04)**
*   **[如何用 conda 使用 python 虚拟环境](https://www.milindsoorya.com/blog/how-to-use-virtual-environment-with-conda)**
*   **[用 python 构建垃圾邮件分类器](https://www.milindsoorya.com/blog/build-a-spam-classifier-in-python)**