# 使用 python 自动发送和安排电子邮件

> 原文：<https://medium.com/analytics-vidhya/automate-and-schedule-sending-of-e-mail-using-python-d26b3e1ff38b?source=collection_archive---------1----------------------->

本文的目的是解释我们如何使用我们的个人或专业电子邮件地址来自动化和安排使用 python 发送电子邮件。

## I .使用 python 创建和发送电子邮件

**第一步:**导入 python 包

```
####################################################################
############# Import packages
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
```