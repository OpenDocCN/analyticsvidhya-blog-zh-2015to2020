# 使用 Python3 和 Selenium 实现简单的 Whatsapp 自动化

> 原文：<https://medium.com/analytics-vidhya/simple-whatsapp-automation-using-python3-and-selenium-77dad606284b?source=collection_archive---------0----------------------->

在我之前的故事里([https://medium . com/@ DeaVenditama/how-I-build-whatsapp-automation-tools-74 f 10 c 93 e 9 b 8](/@DeaVenditama/how-i-build-whatsapp-automation-tools-74f10c93e9b8))我告诉过你一般的步骤怎么做，现在我从技术上用代码解释一下怎么做。我在本教程中使用 Python 3 和 Selenium，并且我假设读者了解 Python 的基础知识。

![](img/ec2d2496f422c9348d38990e9de91151.png)

照片由[张秀坤镰刀](https://unsplash.com/@drscythe?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

首先，我们必须安装 Python 3；你可以从[https://www.python.org/](https://www.python.org/)下载并按照安装说明进行操作。安装 Python3 后，安装 Selenium Automation Framework 来自动化我们以后要做的所有事情。我推荐使用 pip 安装 Selenium。

```
python3 -m pip install Selenium
```

# 硒你好世界

Selenium 成功安装后，要测试它是否正确安装在您的机器上，运行这个 Python 代码并检查是否有任何错误消息。

```
from selenium import webdriver
import time
driver = webdriver.Chrome()
driver.get("http://google.com")
time.sleep(2)
driver.quit()
```

将此代码保存在一个名为 automate.py 或任何 python 文件名的文件中，然后运行它，它将显示谷歌浏览器窗口，并自动转到 google.com。

# 自动化 Whatsapp

```
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import timecontact = "John"
text = "Hey, this message was sent using Selenium"driver = webdriver.Chrome()
driver.get("https://web.whatsapp.com")
print("Scan QR Code, And then Enter")
input()
print("Logged In")inp_xpath_search = "//input[@title='Search or start new chat']"
input_box_search = WebDriverWait(driver,50).until(lambda driver: driver.find_element_by_xpath(inp_xpath_search))
input_box_search.click()
time.sleep(2)
input_box_search.send_keys(contact)
time.sleep(2)selected_contact = driver.find_element_by_xpath("//span[@title='"+contact+"']")
selected_contact.click()inp_xpath = '//div[@class="_2S1VP copyable-text selectable-text"][@contenteditable="true"][@data-tab="1"]'
input_box = driver.find_element_by_xpath(inp_xpath)
time.sleep(2)
input_box.send_keys(text + Keys.ENTER)
time.sleep(2)driver.quit()
```

将上面的代码复制粘贴到 automation.py 中，运行，就会显示一个 Whatsapp 的 Web 界面。扫描 QR 码，等待加载完成，然后在终端/命令提示符下按 enter 键继续发送消息。

这是上面对 Python 代码的解释，

```
contact = "John"
text = "Hey, this message was sent using Selenium"
```

这两行代码是存储联系人姓名和消息的 Python 变量。

```
driver = webdriver.Chrome()
driver.get("https://web.whatsapp.com")
print("Scan QR Code, And then Enter")
input()
print("Logged In")
```

它将打开一个 Whatsapp Web 界面，自动要求您扫描二维码，input()用于暂停该过程，直到您在终端中按下 enter 按钮。

```
inp_xpath_search = "//input[@title='Search or start new chat']"
input_box_search = WebDriverWait(driver,50).until(lambda driver: driver.find_element_by_xpath(inp_xpath_search))
input_box_search.click()
time.sleep(2)
input_box_search.send_keys(contact)
time.sleep(2)
```

inp_xpath_search 变量是一个 xpath 路径，用于查找搜索输入框。搜索框找到后，它会点击搜索框，输入联系人的姓名，然后自动输入。

```
selected_contact = driver.find_element_by_xpath("//span[@title='"+contact+"']")
selected_contact.click()
```

单击在搜索结果中找到的联系人

```
inp_xpath = '//div[@class="_2S1VP copyable-text selectable-text"][@contenteditable="true"][@data-tab="1"]'
input_box = driver.find_element_by_xpath(inp_xpath)
time.sleep(2)
input_box.send_keys(text + Keys.ENTER)
time.sleep(2)
```

找到消息文本框，输入消息，然后自动按 enter 键

```
driver.quit()
```

driver.quit()用来杀死 chromedriver 的实例或者你使用的任何驱动程序，如果你不包含这行代码，这个进程将一直在你的电脑内存中运行。

好了，本教程到此结束，在接下来的故事中，我将向你展示如何将代码包装在 Python 类中，这样你就可以更加动态和可重用地使用它。也许我会涵盖一些新的功能，如收集联系人或消息。谢谢你。