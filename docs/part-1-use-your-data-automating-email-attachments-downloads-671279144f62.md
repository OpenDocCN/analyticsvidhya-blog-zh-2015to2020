# 使用您的数据:自动下载电子邮件附件

> 原文：<https://medium.com/analytics-vidhya/part-1-use-your-data-automating-email-attachments-downloads-671279144f62?source=collection_archive---------14----------------------->

![](img/77b8b836c157872aba42fe36d05a60cc.png)

## 第 1 部分—问题陈述的形成和数据的收集

这是展示如何使用你的数据解决问题的两篇文章中的第一篇。我相信，在应用数据科学技能之前，您不需要解决大问题。有些问题就在我们面前，只要仔细关注和自动化，我们就能解决它们。第一部分介绍如何自动下载电子邮件附件，第二部分介绍如何抓取 pdf 并提取相关数据。请注意，我们不会共享任何数据，因为这些都是财务敏感信息。

![](img/bdc47a3990420f271f3a18cd7c640876.png)

问题概述:最近，我的孩子所在的家庭日托中心的老板通知我，有一笔未付的款项。*杰出？哇！这么多？怎么会这样*她解释说，给我孩子的儿童保育补贴一直在波动，她最近刚刚完成了检查。她给了我一张到目前为止所欠的清单，我向她保证我会检查我的记录并整理出来。

好了，我们开始工作吧。

![](img/2ea9f9b0af4c43a752608f4afbde55d6.png)

问题定义:我欠了多少儿童保育费用？我如何阻止这些付款差异再次发生？

行动方针:从电子邮件中下载权利声明(这提供了使用小时数、Centrelink 支付的儿童保育补贴和父母应支付的未付款项的细目)，并进行一些计算。现在我可以用手来做，通过手动增加和减少数量。但是三个月后会发生什么呢？六个月？我必须重复这个非常手动的过程吗？绝对不方便！！！当然，这必须自动化。

我采用了 OSEMN 数据科学模型的前三个部分(获取数据、清理数据、探索数据)——【http://www.dataists.com/2010/09/a-taxonomy-of-data-science。在本文中，我们将关注第一部分——获取数据。

第一步——获取数据:权利声明以 PDF 文件的形式发送到我妻子的收件箱。多方便啊。我不得不设法收集这些 PDF 文件，将它们合并成一个电子表格，然后进行计算，找出由于儿童保育补贴的波动，我欠了多少欠款。但首先我得拿到 pdf 文件？怎么会？首先想到的是下载每一个，虽然不是很多，但是不方便。必须有一种简化事情的方法。

如果我找到一种自动下载过程的方法，程序会进入邮箱，筛选主题为“权利声明”的邮件，找到这些特定邮件中的附件，并将它们下载到我的电脑上，会怎么样？

Python 是我研究数据科学的工具之一。这是简单，容易和有效的。下面的脚本展示了整个自动化过程。

```
# Import relevant modules
import imaplib
import configparser
import email
from pathlib import Path
```

Imaplib 包有助于连接到电子邮件服务器，特别是用于检索。电子邮件包管理电子邮件。在这种情况下，ConfigParser 帮助检索存储在单独文件中的邮件用户名和密码——您不希望您的电子邮件详细信息出现在脚本中——这肯定是不安全的！Pathlib 有助于文件访问，在我看来，它比 OS 包更容易使用。

好了，模块装载完毕。让我们登录我们的邮箱。首先，我们创建一个包含用户名和密码的文件，并将其保存为*‘something . ini’*。然后，我们使用 configparser 引用该文件，并登录到我们的 gmail 帐户。同样，将我们的详细信息保存在一个单独的文件中的整个想法是出于安全原因——确保没有人能够通过脚本访问我们的登录详细信息。

```
#login details stored in config file and kept separate
config_path = '/home/sam/Everything Python/config_file_childcare.ini'
config = configparser.ConfigParser()
config.read(config_path)
host = 'imap.gmail.com'
user = config.get('gmail','username')
password = config.get('gmail','password') # Connect to the server
print('Connecting to ' + host)
mailBox = imaplib.IMAP4_SSL(host) # Login to your account
 mailBox.login(user, password)
```

此时，您应该会在屏幕上看到一条确认登录成功的消息。如果失败，您会收到“验证失败”错误，您可以更改 gmail 上的设置，允许不太安全的应用程序访问。一旦执行了脚本，您可以随时关闭它。邮箱默认为您的收件箱。当然，您可以访问邮件的其他部分，但这里我将重点讨论我们的问题范围。

好了，我们进去了，让我们去拿那些附件。我是说，这就是我们在这里的目的。

```
mailBox.select()
	searchQuery = '(SUBJECT "Statement of Entitlement")'  _, data = mailBox.uid('search', None, searchQuery)
```

我们在这里搜索我们的邮件。我们很具体，因为我们知道题目。搜索是通过 uid 唯一标识符完成的。每条消息都有一个 UID。建议按 UID 搜索——这是直接从 Python 的 Imaplib 文档中复制的:*“注意，IMAP4 的消息编号随着邮箱的变化而变化；特别是，在 EXPUNGE 命令执行删除操作后，剩余的邮件会重新编号。因此，强烈建议通过 UID 命令使用 UID。https://docs.python.org/3.7/library/imaplib.html*

搜索的输出是一个元组—我们只对元组的第二部分感兴趣，它包含相关的消息(以字节为单位)。我们仍然有一些方法去得到我们的附件。来吧，我们走——快到了。

```
for num, latest_email_uid in enumerate(data[0].split()):  _,box = mailBox.uid('fetch',latest_email_uid,'(RFC822)')  c = email.message_from_string(box[0][1].decode('utf-8'))  for part in c.walk():
            #only interested in the part of the email that has attachment
	        if part.get_content_disposition()!='attachment': 

	            continue
	        file_name = f'{num+1}_{part.get_filename()}'
	        print(file_name)
	        attachment = part.get_payload(decode=True)  file_dir = Path('./Data/Email_Downloads')  file_ = file_dir/file_name
	        if not file_.exists():
	            file_.touch()
	            file_.write_bytes(attachment)
	            print('file created!')
	        else:
	            print(f'{file_} already exists!') mailBox.close()
mailBox.logout()
```

基本上，该脚本所做的是遍历我们的邮件，只查找带有附件的邮件的子部分，并将它们下载到我的计算机上的特定位置。

现在你知道了。电子邮件附件可以自动下载到我的电脑上。下面是我的 github 资源库上完整脚本的链接:[*https://github . com/samukweku/PDF _ Extraction/blob/master/attachment _ downloads . py*](https://github.com/samukweku/PDF_Extraction/blob/master/attachment_downloads.py)

试一试，调整它，打破它，根据你的需要配置它，让我知道你的想法。欢迎反馈和建议。

让我们开始第二部分——提取数据。