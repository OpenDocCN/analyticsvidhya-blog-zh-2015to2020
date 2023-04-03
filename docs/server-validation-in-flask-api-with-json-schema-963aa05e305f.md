# 带有 JSON 模式的 Flask API 中的服务器验证

> 原文：<https://medium.com/analytics-vidhya/server-validation-in-flask-api-with-json-schema-963aa05e305f?source=collection_archive---------2----------------------->

![](img/40e0e9d064ec29ac6dea1f98b130d2fd.png)

Aviv Perets 在 [pexels](https://www.pexels.com/@avivperets) 上拍摄的照片

在 API 世界中，验证意味着检查发送的数据是好是坏。你永远不能完全依赖于只有客户端验证。因为你不知道客户端上发生了什么，所以你不能相信你收到的数据。即使你有一个私有的 API，有人仍然可以向你发送无效的请求。

服务器端验证意味着检查多项内容:

*   预期的属性是什么
*   他们有好的格式/类型吗
*   该属性是必需的吗

# Json 模式

JSON 模式是一种描述任何 JSON 数据实例的方式，就像在我们的 HTTP 请求或响应中发现的那样。

关键字 ***type*** 会把你的属性限制在某个类型。有几个可能的值:

*   线
*   数字
*   目标
*   排列
*   布尔型
*   空

## 数字

```
"my_param": { "type": "number" }
```

您可以使用附加参数确保您的参数在某个范围内:

*   *x* ≥ `minimum`
*   *x*T48`exclusiveMinimum`
*   *x* ≤ `maximum`
*   *x*T49`exclusiveMaximum`

```
"my_param": { 
    "type": "number",
    "minimum": 0,
    "maximum": 100
}
```

## 用线串

```
"my_param": { "type": "string" }
```

还有一些可选参数:

*   minLength(我的字符串可以包含的最小字符数)
*   maxLength(我的字符串可以包含的最大字符数)
*   模式(这是一个正则表达式，如果字符串匹配正则表达式，将被视为有效)

```
"firstname": {
    "type": "string",
    "minLength": 2,
    "maxLength": 50
}
```

## 目标

```
"my_param": { "type": "object" }
```

使用*属性*你可以描述对象内部的所有变量。

```
{
  "type": "object",
  "properties": {
    "number": { "type": "number" },
    "street_name": { "type": "string" },
  },
  "required": ["street_name"]
}
```

*中的一个重要关键字需要*。它将确保在请求中发送一些值。

## **阵列**

```
"my_param": { "type": "array" }
```

如果数组的元素都遵循相同的模式，则可以对它们进行验证。您可以检查数组*最小项*和*最大项的长度。*

```
"my_param": { 
    "type": "array, 
    "minItems*"*: 3,    "items: {
        "type": "number"
    }
}
```

您可以创建复杂元素的列表，并拥有本身包含列表的对象列表…

## **布尔型**

```
"my_param": { "type": "boolean" }
```

这里没有真正的惊喜，值可以是*真*或*假*

## 空

```
"my_param": { "type": "null" }
```

用于表示缺失值，它相当于 python 中的 *None*

## 多种类型

也许一个参数可以有多种可能的类型，例如*字符串*和*无。*当值 send 既可以是空的也可以是字符串。

```
"my_param": { "type": ["string, "null"] }
```

# 用烧瓶验证

## 包裹

有几个 python 包会基于 JSON 模式进行验证。我个人使用[**flask-expects-JSON**](https://pypi.org/project/flask-expects-json/)**。**

要安装它:

```
pip install flask-expects-json
```

## 它是如何工作

这个包只是作为你的 flask 端点的装饰器。

```
schema = {
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "email": { "type": "string" }
  },
  "required": ["email"]
}@app.route('/', methods=['POST'])
@expects_json(schema)
def example_endpoint():
    ...
```

如果收到的请求与模式不符，将会引发错误。并将发送一个 400 错误作为响应。-

## 跳过验证方法

如果您的端点有几个方法，您可以忽略其中一些方法的验证。

```
@app.route('/', methods=['GET', 'POST'])
@expects_json(schema, ignore_for=['GET'])
def example():
    return
```

json-schema 中还有其他有用的关键字。我在这里只讨论了最有用的。