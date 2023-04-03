# 在 PyTest 中注册自定义标记

> 原文：<https://medium.com/analytics-vidhya/registering-custom-markers-in-pytest-cfe6b5620d3c?source=collection_archive---------5----------------------->

![](img/d6332a0abc871fa9f79b03565af74709.png)

[丹尼·米勒](https://unsplash.com/@redaquamedia?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

当在 PyTest 中使用自定义标记时，我们需要在使用之前注册它们。让我们来看看如何。

> 新的 Pytest [检查这个](https://blog.usejournal.com/getting-started-with-pytest-c927946b686)
> 新的标记[检查这个](/@pradeepa.gp/grouping-tests-in-pytest-substrings-and-markers-d145fcd1053d)

在下面的示例文件`test_custom_markers.py`中，我们有一个自定义标记`@pytest.mark.regression`

测试 _ 自定义 _ 标记. py

```
import pytest…
```