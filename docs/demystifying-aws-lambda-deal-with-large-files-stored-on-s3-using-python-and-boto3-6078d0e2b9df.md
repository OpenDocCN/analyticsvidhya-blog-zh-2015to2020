# 使用 AWS Lambda 逐行处理大文件

> 原文：<https://medium.com/analytics-vidhya/demystifying-aws-lambda-deal-with-large-files-stored-on-s3-using-python-and-boto3-6078d0e2b9df?source=collection_archive---------2----------------------->

## 使用 boto3 和 python 使用无服务器 FAAS 功能逐行处理文件，并充分利用它

![](img/47ed48df9969520efcbeeb551121552f.png)

照片由[阿尔弗雷德](https://unsplash.com/@alfredsd?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

想到用于执行工作负载的大型物理服务器，你的脑海中就会浮现出上面的画面。现在考虑购买这些巨大的服务器来处理您的数据，这并不是一个好的选择，对吗？

为什么不利用云中的服务器，在云服务器上运行我们的工作负载呢？好主意，但另一个问题是，现在我们必须管理我们的工作负载，并且还要注意在适当的时候关闭服务器，以避免额外的成本。没有人愿意为不必要的东西付钱。为什么我们不能有一些我们不需要管理的东西呢？为什么我们不能为我们使用的东西付费？为什么我们不能为使用服务器的时间付费？

好吧，这就是无服务器模式。你不想购买巨大的服务器。您不希望因服务器未被利用而被收费。对于特定的工作负载，您只需要特定的内存。走向无服务器是你所有疑问的答案。

无服务器并不意味着你的程序在没有服务器的情况下也能运行，而是当你需要服务器的时候，它会以最低的最优价格提供给你，而且你只需为你的程序被执行的时间付费。所以，从技术上来说，服务器并没有过时，它们只是被抽象化了，所以我们更关注我们的程序，而不是服务器管理。

AWS Lambda 是无服务器的 **FAAS** (功能即服务)，它使您能够在不提供物理服务器或利用云服务器的情况下运行您的程序。

Lambda 函数虽然非常强大，但自身却没有什么限制:

1.  Lambda 函数运行时间不能超过 15 分钟。
2.  Lambda 函数不能使用大于 3GB 的内存

为了从 s3 读取文件，我们将使用 boto3:

λ要点

现在，当我们使用 get_object 读取文件而不是返回完整的数据时，它会返回该对象的 **StreamingBody** 。

```
{
    'Body': **StreamingBody()**,
    'DeleteMarker': **True|False**,
    'AcceptRanges': 'string',
    'Expiration': 'string',
    'Restore': 'string',
    'LastModified': datetime(2015, 1, 1),
    'ContentLength': 123,
    'ETag': 'string',
    'MissingMeta': 123,
    'VersionId': 'string',
    'CacheControl': 'string',
    'ContentDisposition': 'string',
    'ContentEncoding': 'string',
    'ContentLanguage': 'string',
    'ContentRange': 'string',
    'ContentType': 'string',
    'Expires': datetime(2015, 1, 1),
    'WebsiteRedirectLocation': 'string',
    'ServerSideEncryption': 'AES256'**|**'aws:kms',
    'Metadata': {
        'string': 'string'
    },
    'SSECustomerAlgorithm': 'string',
    'SSECustomerKeyMD5': 'string',
    'SSEKMSKeyId': 'string',
    'StorageClass': 'STANDARD'**|**'REDUCED_REDUNDANCY'**|**'STANDARD_IA'**|**'ONEZONE_IA'**|**'INTELLIGENT_TIERING'**|**'GLACIER'**|**'DEEP_ARCHIVE',
    'RequestCharged': 'requester',
    'ReplicationStatus': 'COMPLETE'**|**'PENDING'**|**'FAILED'**|**'REPLICA',
    'PartsCount': 123,
    'TagCount': 123,
    'ObjectLockMode': 'GOVERNANCE'**|**'COMPLIANCE',
    'ObjectLockRetainUntilDate': datetime(2015, 1, 1),
    'ObjectLockLegalHoldStatus': 'ON'**|**'OFF'
}
```

这里可以找到[。](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object)

这个流主体为我们提供了各种选项，比如分块读取数据或逐行读取数据。关于 [StreamingBody](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/response.html) 的所有可用选项，请参考此[链接](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/response.html)。

当我们运行下面的命令时，我们默认读取完整的数据，这是我们需要不惜一切代价避免的。

```
reponse['Body'].read()
```

根据文档，我建议避免使用:

**read( *amt=None* ):** 从流中最多读取 amt 字节。如果省略 amt 参数，则读取所有数据。

而是更喜欢

**ITER _ lines(*chunk _ size = 1024*):**返回一个迭代器，从原始流中产生行。这是通过从原始流中一次读取字节块(大小为 chunk_size ),然后从中产生行来实现的。

**ITER _ chunks(*chunk_size = 1024*):**返回一个迭代器，从原始流中产生 chunk _ size 字节的块。

现在，由于在我们运行 get_object 时没有返回完整的对象，这为 lambda 打开了一个新的可能性世界。现在，我们可以在 step 函数的帮助下链接多个 lambda 函数，也可以通过设置 s3 bucket 事件将值从一个 lambda 传递到另一个 lambda。

这使得数据工程师能够以最低的成本完成许多任务。希望你喜欢这篇文章。

敬请关注更多内容。

参考资料:
【1】[boto 3 文档](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.get_object)

[2] [响应参考文件](https://botocore.amazonaws.com/v1/documentation/api/latest/reference/response.html)