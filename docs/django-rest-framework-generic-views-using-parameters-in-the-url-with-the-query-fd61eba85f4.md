# Django Rest 框架:通用视图，在查询中使用 Url 中的参数

> 原文：<https://medium.com/analytics-vidhya/django-rest-framework-generic-views-using-parameters-in-the-url-with-the-query-fd61eba85f4?source=collection_archive---------5----------------------->

*一名导游*

![](img/dcf8f163b55d69974a06792d05e842c7.png)

Django rest framework 的通用视图是创建 API 端点的一种很好的方式，并且不会在处理数据的技术细节上浪费太多时间。然而，因为视图是为了适应标准使用而内置的(因此得名*通用*)，所以它不能很好地满足定制需求。因此，我们发现自己正在从头开始写观点。

但是如果我们不需要呢？如果我们能够*定制*通用视图会怎么样？

自定义需求的一个例子是创建端点，这些端点使用 URL 中的参数和查询体中的参数。例如，假设我们有一个包含以下端点的图书模型:

```
/books/
/books/<book_id:int>/
```

我们想在这个模型中添加一个图像字段来存储图书封面。我们希望端点保持一致，因此图像端点的设计如下所示:

```
/books/<book_id:int>/image/
```

这个端点需要处理 GET 请求，该请求将返回指定书籍的图像(如果有)。它还处理向图书添加或更改图像的 POST 请求。

因为我们不希望在查询体和端点的 URL 中出现重复的`b` ook_id `,所以我们将创建一个仅包含 image 字段的 serialize，然后使用 URL 中的参数来标识我们正在处理的图书。这就是为什么序列化程序将如下所示:

但是，因为通用视图(generics。创建和通用。RETRIVE)在创建中只使用查询数据，因为`b`ook _ id’在 URL 中而不是在查询中，我们将不得不覆盖创建过程。

但是我们如何在序列化器中获得 book_id 呢？

为此，我们将创建一个通道，将 book_id 从视图传送到序列化程序。我们通过覆盖视图上的 get_serializer_context 方法来实现，并传递带有上下文的 book_id，然后通过覆盖图书的序列化程序上的 validate 方法从另一端获取它。

所以视图会是这样的:

NB。因为我们正在处理一个文件，我们必须设置解析器类来处理，因此:

```
parser_classes = (MultiPartParser, FormParser)
```

序列化程序将会这样结束:

感谢您的阅读，希望对您有所帮助。