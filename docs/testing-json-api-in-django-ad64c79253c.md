# 在 Django 中测试 API 的 JSON 响应

> 原文：<https://medium.com/analytics-vidhya/testing-json-api-in-django-ad64c79253c?source=collection_archive---------0----------------------->

问题:我需要测试 API 端点的 JSON 响应。API 端点遵循 [JSON API 规范](http://jsonapi.org/)，使用`application/vnd.api+json`的`Content-Type`头，而不是内置`[.json()](https://docs.djangoproject.com/en/1.10/topics/testing/tools/#django.test.Response.json)` [方法](https://docs.djangoproject.com/en/1.10/topics/testing/tools/#django.test.Response.json)所需的`application/json`。

解决方案:我找不到比手动将`response.content`解析成 JSON 对象更好的方式了。

```
import json
self.assertEqual(
  json.loads(response.content)['data'],
  []
)
```

注意:如果有某种方法允许`.json()`方法…