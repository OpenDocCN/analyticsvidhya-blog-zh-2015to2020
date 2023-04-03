# 如何用 Python Twint 从 Twitter 上抓取推文

> 原文：<https://medium.com/analytics-vidhya/how-to-scrape-tweets-from-twitter-with-python-twint-83b4c70c5536?source=collection_archive---------0----------------------->

Twint 是一个用 Python 编写的高级 Twitter 抓取工具，允许从 Twitter 抓取 Tweets。

[](https://github.com/twintproject/twint) [## twintproject/twint

### 没有认证。没有 API。没有限制。Twint 是一个用 Python 编写的高级 Twitter 抓取工具，它允许…

github.com](https://github.com/twintproject/twint) 

TWINT 的好处是你**不需要** Twitter 的 API 来让 Twint 工作。Twint 利用 Twitter 的搜索运营商让您:

*   从特定用户那里收集推文
*   抓取与某些话题相关的推文
*   标签和趋势
*   或者从电子邮件和电话号码之类的推文中挑出*敏感*信息。

# 使用 Twint 的一些好处:

*   可以获取几乎所有的推文(Twitter API 限制只能持续 3200 条推文)；
*   快速初始设置；
*   可以匿名使用，无需 Twitter 注册；
*   没有费率限制。

# 先决条件

*   python 3.6
*   Twint 2.1.19
*   虚拟环境

# 创建虚拟环境(VENV)

在你的**终端**或 **cmd** 中键入以下命令:

*   窗子

```
py -m venv venv
```

*   Linux 操作系统

```
python -m venv venv
```

要激活您的 **venv** ，请在您的 **cmd** 或**终端** linux 中键入以下命令:

```
#Windows
    .\venv\Scripts\activate.bat#Linux
    source venv/bin/activate
```

# 安装 Twint

在 venv 中，通过在终端中输入以下命令来安装 Twint。

```
pip install twint
```

请稍等片刻，直到安装过程完成，现在您已经在 venv 上安装了 twint

> ！！确保将它安装在您的 **venv** 中

# 创建刮刀程序

要使用 twint，我们将使用 **python** 语言，安装 Twint 后，您需要导入 Twint。

```
import twint
```

好的，例如，我们将制作一个程序，通过查询“**比特币**”这个词来抓取 **10** 条推文

```
import twint#configuration
config = twint.Config()
config.Search = "bitcoin"
config.Limit = 10#running search
twint.run.Search(config)
```

该程序将只查询“bitcoun”这个词的推文，并检索多达 10 条最近的推文。要创建另一个复杂的命令，您可以添加一些额外的配置。因此，我将使用**英文** tweets 在从*2019–04–29】*到*2020–04–29】*的 **100 条** tweets 中查询单词“**比特币**，并将输出保存为 **json** 格式文件

```
import twint#configuration
config = twint.Config()
config.Search = "bitcoin"
config.Lang = "en"
config.Limit = 100
config.Since = "*2019–04–29*"
config.Until = "*2020–04–29*"
config.Store_json = *True
config.Output = "custom_out.json"*#running search
twint.run.Search(config)
```

那么上述程序的目的是什么，我将逐一解释如下:

*   **搜索** =在这里填写您想要搜索的查询
*   **郎** =你可以指定你想要抓取的推文的语言，关于语言代码你可以在这里看到
*   **限制** =限制被抓取的推文数量
*   **因为** =给出将被删除的 tweet 的日期的具体时间，如果还没有达到该时间的限制，则删除将结束
*   **直到** =像**从**开始但是**直到**命令用来给时间开始刮。Twint 从最大时间到最小时间开始报废，另一个使用它的例子"***2020–01–18 15:51:31***
*   **Store_json** =可以将输出数据以 json 文件的形式保存，值为 ***True*** 或 ***False、*** 也可以将 **CSV** 格式改为“ **Store_csv** ”。
*   **output**=以特定的名称或目录保存输出数据

Twint 中有更多的搜索功能，关于其他配置，您可以查看以下页面:

[](https://github.com/twintproject/twint/wiki/Configuration) [## twintproject/twint

### 变量类型描述-用户名(字符串)- Twitter 用户的用户名…

github.com](https://github.com/twintproject/twint/wiki/Configuration) 

# 结束语

谢谢你把这个教程看完。希望这篇文章能帮助你，下一篇文章再见