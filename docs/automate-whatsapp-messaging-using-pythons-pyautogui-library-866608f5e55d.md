# 使用 Python pyautogui 库自动化 WhatsApp 消息传递——sandy 启发

> 原文：<https://medium.com/analytics-vidhya/automate-whatsapp-messaging-using-pythons-pyautogui-library-866608f5e55d?source=collection_archive---------4----------------------->

![](img/a2e075aed5b7d0e6b1118f412d6c0e36.png)

路易斯·维拉斯米尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

**稍微介绍一下我吧！**

我在 Cognizant 担任数据工程师；在过去的三年里，我一直积极参与黑客马拉松，构建我希望看到的解决方案。在过去两年中赢得了五次黑客马拉松，其中**三次是微软主办的(获得了价值约 6500 美元的奖金)**。本人持有六个 Azure 认证(包括 **DP-203 和 DP-100** )。我几乎所有的周末都用来参加黑客马拉松、开源和构建 Azure。最近在印度钦奈莉拉宫举办的 **Azure 社区大会 2022** 上发表演讲(亚洲最大的 Azure 社区大会)。我还是钦奈地区的 Azure 开发者社区负责人。

你厌倦了在 WhatsApp 上转发消息吗？？这里有一个很酷的方法可以使用 python 来自动化这个过程。你能想到的 python 都有！！

# 介绍

自从我开始了机器学习的旅程，我就一直想把身边的事情自动化。这不是我第一次尝试自动化。我尝试过在朋友生日时自动发送生日祝福(正在开发中)，因为我不太擅长记数字。:D

# 我们开始吧

首先，您需要下载 pyautogui 库。这个库将帮助我们在你的浏览器中自动完成任务。

```
pip install pyautogui
```

# 密码

这将导入所有需要的库

接下来，您需要联系人列表来发送邮件

```
#contact listcontacts = [“Dave”,”Harry”]message = “Hello”
```

您可以在联系人列表中添加任意数量的姓名

**注意:联系人列表中的姓名必须与 Whatsapp 联系人中的姓名完全一致**

# 密码

此方法将在搜索框中输入名称

这个方法将输入消息并发送它

```
# method to find and send message
def click_send_message(msg): 
    x3, y3 = [950,750] 
    moveTo(x3, y3) 
    click() 
    sleep(2) 
    typewrite(msg) 
    press(‘enter’)
```

最后，您需要遍历联系人列表中的姓名，并相应地发送消息

# **运行代码**

```
python main.py
```

你必须在浏览器上登录 Whatsapp 网站，然后才能运行这个文件。运行 python 文件后，切换到浏览器。我已经用 chrome 浏览器测试了代码。如果你得到错误“无法定位搜索栏”尝试改变坐标

# 结论

我希望这能帮助你在 Whatsapp 中发送自动消息，让你的生活更轻松。

跟随我学习更多像这样的酷把戏，并评论你的想法。

感谢您的宝贵时间！！！

# 分叉我的回购报告问题和错误

[](https://github.com/Santhoshkumard11/whatsapp-message-bot) [## santhoshkumard 11/whatsapp-message-bot

### 使用 python Medium Blog =>…

github.com](https://github.com/Santhoshkumard11/whatsapp-message-bot)