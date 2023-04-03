# 使用 python 创建 MailChimp 电子邮件活动

> 原文：<https://medium.com/analytics-vidhya/mailchimp-email-campaign-creation-using-python-7d7825292039?source=collection_archive---------3----------------------->

![](img/421dab860eb3783d693f93a4065d102e.png)

电子邮件营销是与你的电子商务客户沟通的一种有效而强大的方式，因为全世界超过 34%的人使用电子邮件。当我们思考自己的经历时，我们通常在分析了客户数据库后，每周都会收到一些来自不同类型的自动化电子商务网站的通用和个性化电子邮件。

***为什么电子邮件营销对你的业务如此重要？***

*   *它帮助你与你的观众和核心客户保持联系*
*   *你可以实时联系客户，因为大多数人在移动设备上打开电子邮件*
*   与其他互联网信息相比，电子邮件给人的感觉更个性化、更吸引人
*   *就其表现和结果而言，电子邮件营销是最容易衡量的在线营销类型之一*
*   *其实很实惠*

今天，我将介绍使用 MailChimp API 和 Python 创建通用电子邮件活动的步骤。如果我们想发送定制的电子邮件活动，MailChimp API 也提供了该功能。

因为响应的认证是这个过程的第一步，所以我们需要一个 API 密钥和用户名。现在我们将看到如何在 MailChimp 中生成 **API 键**

1.点按您的个人资料名称以展开“帐户”面板，然后选取“帐户设置”

2.点击附加菜单并选择 **API 键**

3.点击**创建密钥**按钮，复制创建的密钥

**请点击链接观看视频:** [**如何使用 Mailchimp**](https://rb.gy/x52cle) 自动化电子邮件活动

GitHub 资源库:[https://GitHub . com/sherangaG/mailChimp-email-creation-python](https://github.com/sherangaG/mailChimp-email-creation-python)

*导入以下库*

```
from mailchimp3 import MailChimp
from string import Template

import newsletter_template # python script for html template
```

*以下函数用于验证响应*

```
MC_API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
MC_USER_NAME = "XXXXXXXXXXXXXXXX"

client = MailChimp(mc_api=MC_API_KEY, mc_user=MC_USER_NAME)
```

我们需要电子邮件列表，他们的名字，姓氏等。发送给电子邮件收件人。因此，我们需要创建名为**“audience”**的变量来存储客户详细信息。

*下面的功能用于创建观众*

```
def audience_creation_function(audience_creation_dictionary):

        audience_creation = ''
        audience_creation_dictionary = audience_creation_dictionary
        print(audience_creation_dictionary)

        audience_list = {
            "name": audience_creation_dictionary['audience_name'],
            "contact":
            {
                "company": audience_creation_dictionary['company'],
                "address1": audience_creation_dictionary['address1'],
                "city": audience_creation_dictionary['city'],
                "state": audience_creation_dictionary['state'],
                "zip": audience_creation_dictionary['zip_code'],
                "country": audience_creation_dictionary['country']
            },
            "permission_reminder": audience_creation_dictionary['audience_name'],
            "campaign_defaults":
            {
                "from_name": audience_creation_dictionary['from_name'],
                "from_email": audience_creation_dictionary['from_email'],
                "subject": "",
                "language": audience_creation_dictionary['language']
            },
            "email_type_option": False
        }

        try:
            audience_creation = client.lists.create(data = audience_list)
        except Exception as error:
            print(error)

        return audience_creation
```

定义了上面的函数后，我们可以通过调用下面的代码来创建受众。这里我们使用**受众 _ 创作 _ 词典**，它包括*受众 _ 姓名、公司、地址 1、城市、州、邮政编码、国家、发件人 _ 姓名、发件人 _ 电子邮件*和*语言。*

```
audience_creation_dictionary = {
    "audience_name" : "ENTER AUDIENCE NAME",
    "company" : "ENTER COMPANY NAME",
    "address1" : "ENTER ADDRESS",
    "city" :  "ENTER CITY",
    "state" : "ENTER STATE", # EX: Western Province
    "zip_code" : "00300",
    "country" : "ENTER COUNTRY", # EX: LK
    "from_name" : "ENTER FROM NAME",
    "from_email" : "ENTER FROM EMAIL",
    "language" : "en"
} 

audience_creation = audience_creation_function(audience_creation_dictionary)
```

现在我们已经创造了观众。但是里面没有联系人。我们需要向观众添加之前创建的联系人

*以下功能用于向受众添加联系人*

```
def add_members_to_audience_function(audience_id, mail_list, client=client):

        audience_id = audience_id
        email_list = email_list

        if len(email_list)!=0:
            for email_iteration in email_list:
                try:
                    data = {
                        "status": "subscribed",
                        "email_address": email_iteration
                    }

                    client.lists.members.create(list_id=audience_id, data ​=data)
                    print('{} has been successfully added to the {} audience'.format(email_iteration, audience_id))

                except Exception as error:
                    print(error)
        else: 
            print('Email list is empty')
```

为了给观众添加成员，我们需要给参数赋值，这些参数是*观众 id* 和*电子邮件列表*

```
audience_id = audience_creation['id']

email_list = ['ENTER EMAIL ADDRESS1',
              'ENTER EMAIL ADDRESS2']

add_members_to_audience_function(audience_id=audience_creation['id'],email_list=email_list)
```

现在，我们已将成员添加到受众中，现在我们将创建电子邮件营销活动。

*以下功能用于创建活动。*

```
def campaign_creation_function(campaign_name, audience_id, from_name, reply_to, client=client):

        campaign_name = campaign_name
        audience_id = audience_id
        from_name = from_name
        reply_to = reply_to

        data = {
            "recipients" :
            {
                "list_id": audience_id
            },
            "settings":
            {
                "subject_line": campaign_name,
                "from_name": from_name,
                "reply_to": reply_to
            },
            "type": "regular"
        }

        new_campaign = client.campaigns.create(data=data)

        return new_campaign
```

*以下代码用于实现带有三个参数的 campaign _ creation _ function】*

```
campaign_name = 'CAMPAIGN NAME'
from_name = 'FROM NAME'
reply_to = 'REPLY MAIL ADDRESS'campaign = campaign_creation_function(campaign_name=campaign_name,
                                      audience_id=audience_creation['id'],
                                      from_name=from_name,
                                      reply_to=reply_to,
                                      client=client)
```

*我已经使用 html 和 CSS 样式为新闻信函活动创建了简单的 HTML 模板，并嵌入到 python 脚本中(*[*newsletter _ template . py*](https://github.com/sherangaG/mailChimp-email-creation-python/blob/master/newsletter_template.py)*)*

```
def customized_template(html_code, campaign_id, client=client):

        html_code = html_code
        campaign_id = campaign_id string_template = Template(html_code).safe_substitute()

        try:
            client.campaigns.content.update(
                    campaign_id=campaign_id,
                    data={'message': 'Campaign message', 'html': string_template}
                    )
        except Exception as error:

            print(error)
```

*下面的代码用于调用 customized_template 函数*

```
html_code = newsletter_template.html_code           
customized_template(html_code=html_code, campaign_id=campaign['id'])
```

*以下功能用于发送电子邮件*

```
def send_mail(campaign_id, client=client):      
        try:
            client.campaigns.actions.send(campaign_id = campaign_id)
        except Exception as error:
            print(error)
```

*调用 send_email 函数*

```
send_mail(campaign_id=campaign['id'])
```

正如我前面提到的，我们能够使用 python 和 MailChimp API 发送个性化的电子邮件活动。如果你在这方面需要任何帮助，请在下面评论

请点击链接浏览关于如何使用 Mailchimp 自动化电子邮件活动的指南