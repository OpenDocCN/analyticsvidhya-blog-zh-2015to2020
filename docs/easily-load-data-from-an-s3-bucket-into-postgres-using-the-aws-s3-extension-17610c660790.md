# 使用 aws_s3 扩展轻松地将数据从 S3 存储桶加载到 Postgres 中

> 原文：<https://medium.com/analytics-vidhya/easily-load-data-from-an-s3-bucket-into-postgres-using-the-aws-s3-extension-17610c660790?source=collection_archive---------2----------------------->

![](img/106028f41c196903780c50f2180966db.png)

[佩德罗·达·席尔瓦](https://unsplash.com/@pedroplus?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/bucket-data?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

在处理数据时，我们经常会发现我们需要将数据从一个地方移动到另一个地方。在许多情况下，如果您使用 AWS，这些数据可能会存放在 S3 的某个地方，但最终需要存放在数据库中。

如果您使用的数据库也在 AWS 上，并且使用 11.1 或更高版本的 Postgres 引擎，您可以轻松利用 aws_s3 扩展来简化从 s3 加载数据。

# 示例场景

我们已经在网站的前端捕获了结构化的事件数据。事件追踪器会将 S3 的一批事件保存在一个我们可以编程访问的桶中。我们希望将这些数据吸收到我们的 Postgres 中，以分析我们捕获的事件。

# 样本数据文件

```
**event_id,event_name,event_value** a9ae1582–501a-11ea-b77f-2e728ce88125,product_view,coffee mug
a9ae17ee-501a-11ea-b77f-2e728ce88125,search,coffee filters
a9ae19e2–501a-11ea-b77f-2e728ce88125,product_view,french press
```

# 步骤 1 —向 Postgres 添加 aws_s3 扩展

```
CREATE EXTENSION aws_s3 
```

# 步骤 2 —在 Postgres 中创建目标表

```
CREATE TABLE events (event_id uuid primary key, event_name varchar(120) NOT NULL, event_value varchar(256) NOT NULL);
```

# 最后一步——将数据加载到 Postgres！

这一步假设您已经有了一个 AWS 访问密钥和一个 AWS 秘密密钥，可以通过编程访问包含事件数据的 S3 存储桶。如果您还没有这些信息，请参考[如何授权 IAM 访问 S3](https://aws.amazon.com/blogs/security/writing-iam-policies-how-to-grant-access-to-an-amazon-s3-bucket/) 和[在哪里可以找到您的 AWS 访问密钥&机密](https://aws.amazon.com/blogs/security/wheres-my-secret-access-key/)

对于 Postgres，我们运行`[aws_s3.table_import_from_s3](https://docs.amazonaws.cn/en_us/AmazonRDS/latest/AuroraUserGuide/AuroraPostgreSQL.Migrating.html#aws_s3.table_import_from_s3)`命令，将我们的 S3 CSV 文件导入到 Postgres 中。

```
SELECT aws_s3.table_import_from_s3(
'POSTGRES_TABLE_NAME', 'event_id,event_name,event_value', '(format csv, header true)',
'BUCKET_NAME',
'FOLDER_NAME(optional)/FILE_NAME',
'REGION',
'AWS_ACCESS_KEY', 'AWS_SECRET_KEY', 'OPTIONAL_SESSION_TOKEN'
)
```

让我们分解上面语句中的参数，以便更好地理解这里发生了什么:

`**POSTGRES_TABLE_NAME**`Postgres 中的表名

`**event_id,event_name,event_value**`这些是表格中的列，以便与数据匹配。数据将按这些列名的顺序加载，因此请确保它们对齐。

`**(format csv, header true)**`根据你的文件格式，你可以在这里定义。在本例中，它是一个 CSV，但也可以是 JSON 或 gzip。还要注意 header true 语句，只有当你的文件有一个你想跳过的头时才应用它。

`**BUCKET_NAME**`数据所在的 s3 存储桶的名称

`**FOLDER_NAME**`(可选)/文件名
如果您的文件位于该存储桶中的一个文件夹内，那么您可以在这里定义文件的完整路径，包括文件夹。

`**REGION**`
S3 水桶遍布 AWS 的许多不同地区。您需要定义您所指向的桶指向哪里(例如美国东部-1)

`**AWS_ACCESS_KEY**` 从 IAM 创建的编程访问键，参见[亚马逊文档](https://aws.amazon.com/blogs/security/writing-iam-policies-how-to-grant-access-to-an-amazon-s3-bucket/)。

`**AWS_SECRET_KEY**`从 IAM 创建的程序化访问键，参见[亚马逊文档](https://aws.amazon.com/blogs/security/writing-iam-policies-how-to-grant-access-to-an-amazon-s3-bucket/)。

`**OPTIONAL_SESSION_TOKEN**`如前所述，这是可选的，只有当你有一个会话令牌附加到连接上时才需要。

# 结论

该扩展允许您避免不必要的数据移动，并提供完全在 AWS 内的解决方案，同时减少手动或编程将数据从 S3 传输到 Postgres 所需的工作量。

当通过 python 与像 [boto3](https://github.com/boto/boto3) 这样的库结合使用时，您可以利用一个简单的 Lambda 函数，用**更少、更高效的代码**来自动化批量数据移动！