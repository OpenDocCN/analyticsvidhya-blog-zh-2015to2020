# 如何使用 RiveScript 创建基于规则的聊天机器人

> 原文：<https://medium.com/analytics-vidhya/how-to-create-a-rule-based-chatbot-by-using-rivescript-b3ca326b264d?source=collection_archive---------10----------------------->

![](img/621c4caf67fc8e7485a8699f7b276b6a.png)

[亚历山大·奈特](https://unsplash.com/@agkdesign?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

什么是聊天机器人？它是一种允许用户通过聊天通信接口，通过 AI(人工智能)或自定义自动规则与之进行交互的服务。用户通过聊天界面与聊天机器人互动。聊天机器人可以提供各种类型的服务。比如在保险行业的应用中，可以提供保单查询、保费查询、缴费查询、保费变更申请、理赔查询等。

随着即时通讯(IM)技术的发展，许多公司已经开始使用 IM 平台作为提供服务的重要渠道。聊天机器人是新一代用户界面，公司可以在公司网站或社交软件上建立聊天机器人，如 Telegram、Line 和 Facebook Messenger，以提供优质的客户服务。

创建聊天机器人很难吗？如果我完全不懂程序，我还能创建自己的聊天机器人吗？简单来说，搭建一个基本的操作聊天机器人并不难。要创造一个“智能”的聊天机器人，需要通过人工智能或机器学习技术来实现。而且市面上有很多工具可以帮助你建立一个具有人工智能的聊天机器人。

RiveScript 是 chatterbots 的脚本语言，与 AIML 非常相似，但匹配模式和条件控制比 AIML 更完整。RiveScript 是一个基于规则的引擎，其中规则是由人类作者通过称为对话流脚本的过程在程序脚本中创建的。这些使用脚本元语言(简称为“脚本”)作为它们的源代码。我将向你展示如何使用 Python 和 RiveScript 创建聊天机器人。

在本教程中，我们将使用一个需要安装的 Python 库。您可以通过如下的`pip install`命令轻松安装 RiveScript:

```
$ pip install rivescript
# --or--
$ easy_install rivescript
```

 [## 人工智能脚本语言—RiveScript.com

### RiveScript 公开了一种简单的纯文本脚本语言，这种语言易于学习并可以快速开始编写。没必要…

www.rivescript.com](https://www.rivescript.com/) 

RiveScript 在 python 2 和 3 中运行良好，安装后创建一个 chatbot.py，导入以下库。让我们来看看我们今天正在构建的聊天机器人应用程序的代码。您将看到不到 20 行代码:

```
import re
from rivescript import RiveScriptbot = RiveScript(utf8=True, debug=False)
bot.unicode_punctuation = re.compile(r'[.,!?;:]')
bot.load_directory("eg/brain")
bot.sort_replies()msg = sys.argv[1]def response(text):
    text = text.encode("utf-8")
    reply = bot.reply("localuser", text)
    reply_text = reply.encode("utf-8").strip()

    return reply_textdef main():        
    input_text = msg
    input_text = input_text.encode("utf8") reply_text = response(input_text)
    print ("Bot: " + str(reply_text)) if __name__ == "__main__":
    main()
```

让我们花点时间对上面的代码做一些了解。

***第 1 行和第 2 行*** 导入 re 和 rivescript 库。

***第 3 行到第 6 行*** 设置 bot 字典。

***第 7 行到第 11 行*** 创建函数调用响应，这是聊天机器人返回给人类的信息。

***第 12 行和第 16 行*** 创建一个函数调用 main，这是程序的主结构。

***第 17 行和第 18 行*** 调用主函数。

独立模式造就了一个优秀的 RiveScript 开发环境。它允许您运行交互式对话，然后使用**:命令**与它们进行交互。这是一组特殊的交互式命令工具，用于在开发和调试阶段测试和调试您的对话框。

您可以使用存储为普通文本文件的脚本来定义 bot 的对话流。这比其他聊天工具使用的方法简单多了。其他聊天工具通常使用基于浏览器的用户界面、JSON 或 XML。将脚本编写为文本文件可以让您完全控制对话流，并使使用后端脚本和工具处理和升级会话代码变得更加容易。

要创建聊天脚本，您需要在下创建文件。/eg/brain，我会创建一个名为 chat.rive 的文件。

下面是 RiveScript 脚本文件的样子:

```
+ (Hi)
- Hello!
- Hi, how are you?+ (What is your name?)
- (I am Pepper)+ (Good Morning)
- Morning
```

事实上，有许多复杂的函数可以放入上标脚本中，就像下面的例子:创建一个脚本来回答现在是什么时间。

```
> object askTime python
    import time 

    now_hour = int(time.strftime("%H"))
    now_min = (time.strftime("%M"))if (now_hour > 12):
        ampm = "pm"
        now_hour = now_hour - 12
    else:
        ampm = "am"output = str(ampm) + str(now_hour) + str(now_min) 

    return output
< object+ (|*)(What time is it now?)(|*)(what time now?)(|*)
- <call>askTime</call>
```

将所有代码保存到文件中后，启动命令提示符并运行以下命令:

```
>python chatbot.py "What is your name?"
>Bot:My name is Pepper
```

除了使用 RiveScript 创建基于规则的聊天机器人之外，它还可以与微软的 Q&AMaker 或谷歌的 Dialogflow 相结合，创建一个适合自己的聊天机器人

感谢阅读！如果你喜欢这篇文章，请通过鼓掌来感谢你的支持(👏🏼)按钮，或者通过共享这篇文章让其他人可以找到它。

最后，我希望你能学会如何创建一个基本的基于规则的聊天机器人。你也可以在 GitHub 库上找到完整的项目。