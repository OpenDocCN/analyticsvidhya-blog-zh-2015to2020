# 如何发送一个 CSV 文件从 S3 到红移与 AWS Lambda 函数

> 原文：<https://medium.com/analytics-vidhya/how-to-send-a-csv-file-from-s3-into-redshift-with-an-aws-lambda-function-953bf347dddb?source=collection_archive---------1----------------------->

# 介绍

如今，自动化是必须的，云工作也不例外，作为数据工程师，我们需要掌握将数据移动到任何需要的地方的技能，如果你想知道如何像数据专家一样在日常生活中面对 AWS 工具，这篇文章就是为你准备的。

# 逐步地

收集数据后，下一步是设计一个 ETL，以便在您想要将数据移动到 Amazon Redshift 等分析平台之前提取、转换和加载数据，但在这种情况下，只有我们将使用 for AWS free tier 将数据从 S3 移动到 Redshift 集群。

为了做到这一点，我尝试按如下方式处理研究案例:

1.  创建一个 S3 桶。
2.  创建一个红移星团。
3.  从 DBeaver 或者随便什么连接到 Redshift。
4.  在数据库中创建一个表。
5.  用 Python 创建一个需要依赖关系的虚拟环境。
6.  创建你的 Lambda 函数。
7.  有人上传数据给 S3。
8.  查询您的数据。

![](img/964539db0b0d047a7203422340c2ba97.png)

我们开始吧！！

稍后，您已经完成了第 1 步和第 2 步，让我们借助 SQL 客户端[***【DBeaver】***](https://dbeaver.io/)*或任何您想要的东西来连接到我们的数据库，为此我们需要记住来自红移集群配置的以下数据:*

```
*HOST = "xyz.redshift.amazonaws.com"
PORT = "5439"
DATABASE = "mydatabase"
USERNAME = "myadmin"
PASSWORD = "XYZ"
TABLE = "mytable"*
```

*![](img/1b7d01f9f5c21b534c074f42d69d738f.png)*

*连接到数据库*

*现在，当我们连接到数据库时，让我们创建一个新表*

```
*CREATE TABLE mytable (
id      INT4 distkey sortkey,
col 1     VARCHAR (30) NOT NULL,
col 2         VARCHAR(100) NOT NULL,
col 3 VARCHAR(100) NOT NULL,
col 4        INTEGER NOT NULL,
col 5  INTEGER NOT NULL,
col 6           INTEGER NOT NULL);*
```

*对于本教程，我们的 Lambda 函数将需要一些 Python 库，如[***Sqalchemy***](https://pypi.org/project/SQLAlchemy/)***，***[***psycopg 2***](https://pypi.org/project/psycopg2/)***，*** ，所以您需要在压缩之前用 Python 和 Lambda 脚本创建一个虚拟环境。您将上传到 AWS 的 zip 文件。*

*此时，将 Lambda 函数配置到 AWS 中所需要的只是一个 Python 脚本，并在每次有人将新对象上传到 S3 存储桶时触发您的 Lambda，您需要配置以下资源:*

1.  *上传您的 lambda _ function . zip(***Python 脚本和依赖关系*** )并使用下面的代码示例将数据发送到 redshift `lambda_function.py`。*
2.  *将一个 IAM 角色附加到 Lambda 函数，该函数授予对`AWSLambdaVPCAccesExcecutionRole`的访问权限*
3.  *对于这种情况，您需要在 Lambda 函数或任何其他函数中添加 VPC 默认值。*
4.  *添加环境变量“ *CON* 和“ *Table**

```
*CON = "postgresql://USERNAME:PASSWORD[@clustername.](mailto:Datexland686@datexland-129-s3.c7e9zmkabpjl.us-east-1.redshift.amazonaws.com)xyz.redshift.amazonaws.com:5439/DATABASE"
Table = "*mytable*"*
```

*5.创建一个[***【S3】事件通知***](https://docs.aws.amazon.com/AmazonS3/latest/dev/NotificationHowTo.html) ，每当有人上传一个对象到你的 S3 桶时，它就调用 Lambda 函数。*

*6.您可以配置超时时间≥ 5 分钟。*

```
*import sqlalchemy 
import psycopg2
from sqlalchemy import create_engine 
from sqlalchemy.orm import scoped_session, sessionmaker
from datetime import datetime,timedelta
import os def handler(event, context): for record in event['Records']:

      S3_BUCKET = record['s3']['bucket']['name']
      S3_OBJECT = record['s3']['object']['key']

    # Arguments
    DBC= os.environ["CON"]
    RS_TABLE = os.environ["Table"]
    RS_PORT = "5439"
    DELIMITER = "','"
    REGION = "'us-east-1' " # Connection
    engine = create_engine(DBC)
    db = scoped_session(sessionmaker(bind=engine)) # Send files from S3 into redshift
    copy_query = "COPY "+RS_TABLE+" from 's3://"+   S3_BUCKET+'/'+S3_OBJECT+"' iam_role 'arn:aws:iam::11111111111:role/youroleredshift' delimiter "+DELIMITER+" IGNOREHEADER 1 REGION " + REGION # Execute querie
    db.execute(copy_query)
    db.commit()
    db.close()*
```

*在您准备将 CSV 文件上传到您的 S3 存储桶之前，请记住您已经首先创建了一个表，因此在您实现了 lambda 函数并正确配置它之后，您可以将数据上传到 S3，并转到 DBeaver 来查询您的表中的数据。*

*![](img/98293868a1a9b1793283cc2f48108f24.png)*

*执行 Lambda 函数后上传的数据*

# *摘要*

*AWS Lambda 是一种自动化流程的简单方法，但我们需要了解哪个时刻不能使用它，例如，AWS Lambda 有 6MB 的有效负载限制，因此以这种方式迁移非常大的表是不实际的。*

*另一方面，使用这项服务的主要优势是，这是一个无服务器的整体解决方案！！，所以不需要管理任何 EC2 实例。*

*谢谢你读到这里。如果你觉得这篇文章有用，喜欢并分享这篇文章。有人也会觉得它很有用，为什么不邀请我喝杯咖啡呢？*

*[![](img/7b4a8b0ba9a8213d41096ed8ed319752.png)](https://www.paypal.com/donate/?hosted_button_id=GBVXVLXMETRHE)

**用 PayPal 捐款**👆* 

*跟我来👉 [**领英**](https://www.linkedin.com/in/alexanderbolano/)*

*跟我来👉 [**推特**](https://twitter.com/Alex_bonella)*

*联系人:alexbonella2806@gmail.com*