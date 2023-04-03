# 如何用 Python 从 Gmail 中提取邮件和附件

> 原文：<https://medium.com/analytics-vidhya/how-to-extract-emails-and-attachments-from-gmail-with-python-f694f297818b?source=collection_archive---------1----------------------->

第一个中等故事，非常精彩。这里有一个代码为[https://github.com/andreiaugustin/gmail_extractor](https://github.com/andreiaugustin/gmail_extractor)的 TLDR

![](img/0653670fe90fa933cc18f811cf1c963d.png)

由 [Webaroo](https://unsplash.com/@webaroo?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

我最近开发了一个新的支持系统，其中一个目标是将存储在 Gmail 邮箱中的旧数据转移到数据库中。

主要用 C++工作，我很清楚我不能浪费超过一个下午的时间来提取数据，我当然不想用 C++来做。因此，Python 是这项任务的下一个合乎逻辑的选择。

所以我把这个简单的工具放在一起，它提取一些信息，如主题、发件人、日期和邮件正文，并把所有这些信息转换成一种方便的 JSON 格式。此外，这也将提取电子邮件的所有附件。最后，它会将所有内容放入一个以电子邮件的 UID 命名的文件夹中。

逐步构建一切:

首先，创建一个 Python 文件，例如 gmail_extractor.py，让我们开始导入所需的模块。

```
import imaplib
import os
import email
import sys
import json
```

接下来，让我们创建一个满足我们所有需求的类。我知道大多数 Python 开发人员只会为此使用函数，并简单地在 __main__ 中调用它们。然而，我喜欢在栈底保留 __main__ 作为额外的控制级别，以防我需要向程序中添加内容。

我们将调用类 GMAIL_EXTRACTOR，并从定义两个函数开始，helloWorld 和 initializeVariables。helloWorld 只是个人接触，想跳过就跳过:)

我想在 Python 中初始化变量的原因是因为我来自 C++背景，当我阅读一些代码时，我喜欢看到一个类的成员，以便快速找出我将需要什么来将那个类集成到我的代码库中。

```
class GMAIL_EXTRACTOR():
    def helloWorld(self):
        print("\nWelcome to Gmail extractor,\ndeveloped by A. Augustin.") def initializeVariables(self):
        self.usr = ""
        self.pwd = ""
        self.mail = object
        self.mailbox = ""
        self.mailCount = 0
        self.destFolder = ""
        self.data = []
        self.ids = []
        self.idsList = []
```

接下来，让我们定义一个函数来获取用户的登录详细信息:

```
def getLogin(self):
    print("\nPlease enter your Gmail login details below.")
    self.usr = input("Email: ")
    self.pwd = input("Password: ")
```

如你所见，我们只从用户那里得到电子邮件和密码，并将它们保存在 usr 和 pwd 类变量中。

处理登录本身的函数:

```
def attemptLogin(self):
    self.mail = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    if self.mail.login(self.usr, self.pwd):
        print("\nLogon SUCCESSFUL")
        self.destFolder = input("\nPlease choose a destination folder in the form of /Users/username/dest/ (do not forget trailing slash!): ")
        if not self.destFolder.endswith("/"): self.destFolder+="/"
        return True
    else:
        print("\nLogon FAILED")
        return False
```

这里我们硬编码 Google 的 IMAP 地址和端口。他们不太可能改变，所以我们很好。接下来，我们尝试登录并向用户询问目标文件夹，否则我们只打印登录失败并返回 False，这样我们可以在抽象的上一层处理失败。

从用户处获取邮箱名称并将其“选择”为我们的工作邮箱的功能:

```
def selectMailbox(self):
    self.mailbox = input("\nPlease type the name of the mailbox you want to extract, e.g. Inbox: ")
    bin_count = self.mail.select(self.mailbox)[1]
    self.mailCount = int(bin_count[0].decode("utf-8"))
    return True if self.mailCount > 0 else False
```

这将允许我们选择我们希望使用的邮箱，并跟踪邮箱中有多少电子邮件。如果电子邮件的数量为 0 或更少，我们也返回 False，这样抽象的上一层可以处理当邮箱为空时应该发生的事情。在这个工具中，我只是在上面两种情况下使用 sys.exit()-ing，虽然可以实现一些错误消息来使最终用户更容易使用它，但归根结底，这是开发人员的一个工具。

另一个搜索邮箱的功能:

```
def searchThroughMailbox(self):
    type, self.data = self.mail.search(None, "ALL")
    self.ids = self.data[0]
    self.idsList = self.ids.split()
```

这个功能就不多说了。它搜索邮箱中的所有电子邮件，保存 id，并列出 id 列表。

另一个问题是询问用户是否希望继续提取过程:

```
def checkIfUsersWantsToContinue(self):
   print("\nWe have found "+str(self.mailCount)+" emails in the mailbox "+self.mailbox+".")
   return True if input("Do you wish to continue extracting all the emails into "+self.destFolder+"? (y/N) ").lower().strip()[:1] == "y" else False
```

现在，我们将致力于我们的主要功能。如果你愿意，我们的引擎。这也许可以被分成更小的功能，但我只是想在一个下午得到一些工作。让我们称这个函数为 parseEmails:

```
def parseEmails(self):
    jsonOutput = {}
        for anEmail in self.data[0].split():
        type, self.data = self.mail.fetch(anEmail, '(UID RFC822)')
        raw = self.data[0][1]
        raw_str = raw.decode("utf-8")
        msg = email.message_from_string(raw_str) jsonOutput['subject'] = msg['subject']
        jsonOutput['from'] = msg['from']
        jsonOutput['date'] = msg['date']

        raw = self.data[0][0]
        raw_str = raw.decode("utf-8")
        uid = raw_str.split()[2]
        # Body #
        if msg.is_multipart():
            for part in msg.walk():
                partType = part.get_content_type()
                ## Get Body ##
                if partType == "text/plain" and "attachment" not in part:
                    jsonOutput['body'] = part.get_payload()
                ## Get Attachments ##
                if part.get('Content-Disposition') is None:
                    attchName = part.get_filename()
                    if bool(attchName):
                        attchFilePath = str(self.destFolder)+str(uid)+str("/")+str(attchName)
                    os.makedirs(os.path.dirname(attchFilePath), exist_ok=True)
                    with open(attchFilePath, "wb") as f:
                        f.write(part.get_payload(decode=True))
        else:
            jsonOutput['body'] = msg.get_payload(decode=True).decode("utf-8") # Non-multipart email, perhaps no attachments or just text. outputDump = json.dumps(jsonOutput)
        emailInfoFilePath = str(self.destFolder)+str(uid)+str("/")+str(uid)+str(".json")
        os.makedirs(os.path.dirname(emailInfoFilePath), exist_ok=True)
        with open(emailInfoFilePath, "w") as f:
            f.write(outputDump)
```

这里发生了很多事情。我们首先获取电子邮件及其 uid，对其进行解码，然后遍历其各个部分以获取正文和附件，然后将它们保存在每封电子邮件的特定文件夹中。

现在，为了结束这堂课，我们写出 __init__:

```
def __init__(self):
    self.initializeVariables()
    self.helloWorld()
    self.getLogin()
    if self.attemptLogin():
        not self.selectMailbox() and sys.exit()
    else:
        sys.exit()
    not self.checkIfUsersWantsToContinue() and sys.exit()
    self.searchThroughMailbox()
    self.parseEmails()
```

结束我们的节目，我们的主要节目:

```
if __name__ == "__main__":
    run = GMAIL_EXTRACTOR()
```

就是这样！这可以做得更好，但最终它完成了工作，我不用花超过几个小时的时间，所以每个人都很高兴:)

/*更新*/

如果您发现自己遇到了一些 UnicodeDecodeError 异常问题，那是因为您可能正在解析的电子邮件具有不同的编码，即发件人可能正在使用 Outlook。我们可以通过将转换为 UTF-8 的代码封装在 try/except 块中来解决这个问题，并尝试解码 ASCII 和 ISO-8859–1，如下所示:

```
try:
    raw_str = raw.decode("utf-8")
except UnicodeDecodeError:
    try:
        raw_str = raw.decode("ISO-8859-1") # ANSI support
    except UnicodeDecodeError:
        try:
            raw_str = raw.decode("ascii") # ASCII ?
        except UnicodeDecodeError:
            pass
```