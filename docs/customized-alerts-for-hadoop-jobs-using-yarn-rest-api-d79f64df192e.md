# 使用 Yarn Rest API 为 Hadoop 作业定制警报

> 原文：<https://medium.com/analytics-vidhya/customized-alerts-for-hadoop-jobs-using-yarn-rest-api-d79f64df192e?source=collection_archive---------31----------------------->

根据 Cloudera，Ambari 是一个完全开源的管理平台，用于配置、管理、监控和保护 Apache Hadoop 集群。Apache Ambari 消除了操作 Hadoop 的猜测。但是每个用户在 Ambari 上工作时都会遇到的一个问题是，它不提供作业级警报。因此，在 Prod Hadoop 集群中工作时，一些作业可能会运行超过预期时间，或者对资源的请求超过阈值。如果有很多工作，那么找到导致资源紧缺的具体工作就变得很困难。用户可能需要仔细检查每项工作，以确定根本原因。

这个问题可以通过使用自定义代码和调用 Yarn 提供的 rest API 来解决。下面是利用 yarn-rest api 获取作业细节的简单代码示例。您可以进一步定制您的代码，以获得分配的资源和其他详细信息。

# 先决条件

*   SendGrid 帐户/Gmail 帐户
*   Json(解析来自 yarn rest-api 的 json 响应)，用于打开和读取 URL 的 urlib 包
*   电子邮件和 smtp 库发送电子邮件。

下面是可用于获取 hadoop 作业警报的 python 代码。要了解更多信息，您可以点击 [**链接。**](http://thelearnguru.com/customized-alerts-for-hadoop-jobs-using-yarn-rest-api/)

```
import json, urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

#Email function
def send_email(test):
    loginuser = "send grid user"
    loginpassword="send grid password"
    fromaddr="source email "
    toaddr = "target email id"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Alert"
    body = "The list of long running job(s) are " + test
    msg.attach(MIMEText(body, 'plain'))
    server = smtplib.SMTP('smtp.sendgrid.net', 587)
    server.starttls()
    server.login(loginuser,loginpassword)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

link="http://<rm host name>:8088/ws/v1/cluster/apps?states=RUNNING"
#Mention your threshold time in milliseconds
Prescribed_limit=6000 
with urllib.request.urlopen(link) as response:
    result=json.loads(response.read().decode('utf8'))
# Below to parse the json
for  jobs in result['apps']['app']:
    if jobs['elapsedTime']>prescribed_limit:
         send_email(str("\nApp Name: {}".format(jobs['name']) +" with Application id: {}".format(jobs['id'])+" running for {} hours".format(round(jobs['elapsedTime']/1000/60/60)))) 
```

礼貌:Subash Konar