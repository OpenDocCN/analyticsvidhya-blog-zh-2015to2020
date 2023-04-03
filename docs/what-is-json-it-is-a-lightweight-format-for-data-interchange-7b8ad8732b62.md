# JSON 是什么？这是一种轻量级的数据交换格式。

> 原文：<https://medium.com/analytics-vidhya/what-is-json-it-is-a-lightweight-format-for-data-interchange-7b8ad8732b62?source=collection_archive---------10----------------------->

## 让我们看看 JSON 是如何提高网络交互性的。

![](img/4720a29cd520dd5688a8cf53cc0825fc.png)

由 [Szabo Viktor](https://unsplash.com/@vmxhu?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# JSON 是什么？

**JSON** 代表**J**ava**S**script**O**object**N**rotation。它是一种轻量级的基于文本的格式，用于交换数据。它在 web 应用程序中非常常用。它类似于 JavaScript object literal 语法，但是除了 JavaScript 之外，许多其他编程环境都具有使用 JSON 的能力，这使得 JSON 独立于编程语言。“工作”的意思是生成和读取 JSON 数据。

JSON 建立在两种结构之上；名称/值对的集合和有序的值列表，如数组、列表等。我们可以在 JSON 中包含对象、字符串、数字、布尔值、数组和 null。

让我们以一个 JSON 对象为例。JSON 对象意味着名称/值对的无序列表。`{ “name” : “value” }`

```
{ "name" : "Kavindya",
  "age" : 25,
  "address" : {
        "city" : "Galle",
        "country": "Sri Lanka"
  },
  "hobbies" : [ "watching movies", "reading books" ],
  "isStudent" : true
}
```

我们可以将这些数据赋给一个名为 personDetails 的变量。然后，我们可以使用人员详细信息.姓名、人员详细信息.地址、人员详细信息.地址.国家、人员详细信息.爱好等来访问这些数据。

那我们来取一个数组。这是一个有序的值列表。`[ value1, value2 ]`该值可以是字符串、数字、数组、对象、布尔值或空值。

```
[
  { "name" : "Kavindya",
    "age" : 25,
    "address" : {
          "city" : "Galle",
          "country": "Sri Lanka"
    },
    "hobbies" : [ "watching movies", "reading books" ],
    "isStudent" : true
  },
  { "name" : "Dakota",
    "age" : 24,
    "address" : {
          "city" : "Colombo",
          "country": "Sri Lanka"
    },
    "hobbies" : [ "gardening", "playing tennis" ],
    "isStudent" : false
  }
]
```

让我们将这个数组赋给变量 personDetails。然后我们可以使用数组索引来访问数组中的数据。以 personDetails[0]为例。获取值“Kavindya”的名称。当您使用 index 1 时，您将获得关于 Dakota 的详细信息，因为它是数组中的第二条记录。

JSON 忽略 JSON 元素之间的空白，因此我们可以在 JSON 中组织数据，在名称/值对之间留出足够的空间，这样我们就更容易阅读。

# JSON 为什么重要？

随着 AJAX 的到来，我们可以

> 从 web 服务器读取数据—在网页加载后
> 
> 更新网页而不重新加载页面
> 
> 在后台向 web 服务器发送数据

因此，对于网站来说，在不延迟页面渲染的情况下，更快速、更异步地加载数据变得非常重要。JSON 实现了比 XML 更快的可访问性和内存优化。所以 JSON 经常和 AJAX 一起使用。

# 如何在 JavaScript 中使用 JSON 字符串

我们制作 JavaScript 变量。但是经常我们需要把 JavaScript 变量转换成 JSON 数据，把 JSON 数据转换成 JavaScript。

JavaScript 内置方法 JSON.stringify()可用于从 JavaScript 变量输出 JSON 字符串。

```
var data = {
    "id" : 1,
    "username" : "Alice",
    "subjects" : [ "Maths", "Science" ]
} 
```

使用 JSON.stringify(data ),我们可以生成下面的输出。

```
{ "id": 1, "username": "Alice", "subjects": [ "Maths", "Science" ] }
```

要将 JSON 字符串解析成 JavaScript，我们可以使用内置方法 JSON.parse()。它从 JSON 字符串输出一个 JavaScript 对象或数组。

```
var jsonData = { "name": "Alice", "age": 25 };var obj = JSON.parse(jsonData);
```

现在我们可以用 JavaScript 中的`obj.name`和`obj.age`来访问姓名和年龄值。

# 如何在 Java 中使用 JSON

JSON-Java 也称为 org.json，是允许 JSON 数据处理的 Java 库之一。这个库提供了各种可用于解析、操作和转换 JSON 数据的类。

您可以在这里找到来自 Baeldung 的关于 JSON-Java 的完整介绍。[T3【https://www.baeldung.com/java-org-json】T5](https://www.baeldung.com/java-org-json)

我希望这篇文章能帮助您对 JSON 有一个基本的了解。

编码快乐！