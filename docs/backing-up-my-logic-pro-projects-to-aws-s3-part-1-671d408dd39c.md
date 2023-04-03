# 将我的 Logic Pro 项目备份到 AWS S3(第 1 部分)

> 原文：<https://medium.com/analytics-vidhya/backing-up-my-logic-pro-projects-to-aws-s3-part-1-671d408dd39c?source=collection_archive---------11----------------------->

![](img/2b1f436a0a148df1a6e98dab113e4873.png)

一时兴起，只是为了玩玩 Python 和 AWS，我想到了构建一个脚本来将我的 Logic Pro 项目备份到 AWS S3。该脚本的前提是，它将检查我的 Macbook 上的目录，并上传自上次文件上传到 S3 以来被修改过的任何文件。这将通过 cron 作业运行(是的，只有当我的 Macbook 处于唤醒状态时才会运行)。不管怎样，我们开始吧。

至于我如何构建它，一些先决条件:

*   PyCharm 或任何 IDE
*   Python 3.0+版本
*   AWS 帐户([自由层](https://aws.amazon.com/free/)应足以进行测试)
*   对 Python/编程有基本了解

这篇文章不会像你在我的 GitHub(即将发布)上看到的那样，详细介绍每一个代码片段，但是会介绍我组装所需组件的步骤。

# 目标

将本地文件夹中的文件与 S3 和 zip 中的相应文件进行比较，如果本地文件晚于 S3 的文件(上次修改日期时间)，则上传本地文件

# 第一部分

在实现这一点的第 1 部分中，我们将简单介绍如何通过 Python 连接到 S3，以及如何获取最后修改的日期时间，以便稍后进行比较。

因此，我们将浏览这些步骤的代码片段，或者您可以跳过所有这些，只需在 GitHub(即将推出)上查看代码。

## 连接到 S3

如何连接到 S3 自动气象站？幸运的是，AWS 为 Python 提供了一个名为 boto3 的 SDK，所以您只需安装这个模块，然后将其导入到您的脚本中。

我使用的是 Python 3.7 和 pip3，所以我的命令是

```
pip3 install boto3
```

导入以下内容:

```
 # import AWS modules
import boto3
import logging
from botocore.exceptions import ClientError 
```

您还需要安装 aws cli 模块(aws 命令行界面),所以继续使用

```
pip3 install aws cli
```

现在，您需要配置这个 aws cli 工具来连接到您的 aws 帐户，因此您应该有一个用于 cli 的 IAM 用户和组。转到 AWS 控制台中的 IAM 服务来创建它。

## 为 AWS CLI 创建 IAM 用户和组

首先，创建一个组并附加 AdministratorAccess 策略。然后，创建一个用户，并将该用户分配到刚刚创建的组中。继续到向导的最后，您将看到一个按钮，用于下载包含您的凭据的 CSV。这些将是您用来配置 aws cli 的凭据。

准备好访问密钥 ID 和秘密访问密钥，并在终端中键入

```
aws configure 
```

然后会要求您输入密钥 ID 和访问密钥，所以只需粘贴 CSV 中的值。还会要求您输入地区名称(输入您的 S3 存储桶所在的地区)。最后一个输入字段询问输出格式(您可以按 Enter 键)。

太好了！现在，您的 aws cli 应该配置为连接到您的 AWS S3 存储桶。

## 创建 S3 客户端和资源

因此，接下来我们需要创建一个客户端和一个资源来访问 S3 方法(客户端和资源根据用途提供不同的方法，你可以在这里阅读更多的)。

```
 # CREATE CLIENT AND RESOURCE FOR S3
s3Client = boto3.client(‘s3’)
s3Resource = boto3.resource(‘s3’)# object for all s3 buckets
bucket_name = ‘’ # NAME OF BUCKET GOES HERE, HARD CODED FOR NOW# CREATE BUCKET OBJECT FOR THE BUCKET OF CHOICE
bucket = s3Resource.Bucket(bucket_name) 
```

## 检索对象的上次修改日期时间(上传时间)

好，那么需要做什么来检索对象的最后修改日期时间呢？

我编写了一个方法来遍历 bucket 中的对象，并调用一个 AWS 方法来获取上次修改时间，如下所示(是的，这也可以用其他方式编写)。

`bucket.objects.all()`返回一个可以迭代的集合(点击阅读更多关于集合的内容[)。](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/collections.html)

所以你可以看到我开始循环，并且只在 S3 对象包含 *Logic_Projects* 和*时调用 **last_modified** 方法。压缩键名中的*。

如果你不熟悉，关键是 S3 如何识别物体。我正在寻找 *Logic_Projects* ，因为我用那个名字创建了一个文件夹，并且也在检查*。zip* 以防我不小心上传了文件夹里别的东西(但是这个脚本只会上传 zip 到那个文件夹，所以只是安全检查)。

如果它通过了这些检查，那么我继续进行日期时间转换。

你会注意到我在这个循环中调用了另外两个方法，叫做 **utc_to_est** 和 **stamp_to_epoch** 在我最终返回一个值之前。这是为了以后更容易进行日期时间比较。 **last_modified** 方法以 UTC 格式返回一个日期，因此它看起来像这样:*2019–12–07 19:47:36+00:00*。

当我比较修改时间时，我宁愿只比较数字。AWS 返回的日期采用 UTC，也称为 GMT(格林威治标准时间)。

所以我添加了一个函数来将这个日期时间转换为东部标准时间，如下所示:

现在我们在正确的时区，我想把它转换成一个数字，所以我把它转换成纪元时间，这是自 1970 年 1 月 1 日 00:00:00 UTC 以来经过的秒数(为什么纪元时间是这个日期不在本文的范围内)。

好了，第 1 部分到此结束！在这篇文章中，我们**通过 Python (boto3)** 连接到 AWS，并且我们已经创建了一个方法**来获取一个对象在 S3 的最后修改时间(从 Epoch 到 EST 的秒数)。**

在第 2 部分中，我们将比较文件在 S3 的最后修改时间和该文件在本地的最后修改时间。