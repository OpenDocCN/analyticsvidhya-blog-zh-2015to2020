# JSON(JavaScript 对象符号)入门

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-json-javascript-object-notation-38b7a9e0380?source=collection_archive---------14----------------------->

![](img/dc664d383069e66356b8460b38ac6a04.png)

JSON(**J**ava**S**script**O**object**N**rotation)是一种完全独立于语言的存储格式，用于存储和传输数据。这是一个非常重要的话题，因为我们从外部 API 获取的数据通常由 JSON 格式的元素数组组成。

JSON 的语法与 Object literal 语法非常相似，也是由名称、值对组成的。但是在这里，两个名称以及值都用引号括起来。

让我们看看下面的例子:

```
//Object literals syntax
let details = {
      firstName : "John",
      lastName : "Adams",
      age : 27
}

//JSON syntax
{
      "firstName" : "Mike",
      "lastName" :  "Bush",
      "age" : 25
}
```

据信，在前几年，XML 格式被广泛使用，它具有围绕数据的标签。上述 XML 格式 JSON 数据表示如下

```
<details>
     <firstName>Mike</firstName>
     <lastName>Bush</lastName>
     <age>25</age>
</details>
```

如您所见，与 JSON 相比，XML 格式是冗长的，例如，对于单个值“Mike ”,名字“firstName”在开始和结束标记中重复了两次，这是完全不必要的。此外，JSON 可以被解析为对象文字，这使得处理起来更快。

JSON 如此受欢迎，以至于 JavaScript 都能理解它，而且它内置了从 JSON 到 object literal 的转换函数，反之亦然。

*   Javascript 提供了一种将数据从对象文字格式转换成 JSON 格式的方法

```
const objectData = { firstName : "Mike", lastName : "Bush" } const JSONdata = JSON.stringify(objectData) console.log(JSONdata)const JSONdata = '{ "firstName" : "Mike", "lastName" : "Bush"}'; const ObjectData = JSON.parse(JSONdata) console.log(ObjectData)
```

*输出:-*

```
{"firstName":"Mike","lastName":"Bush"}
```

*   还有一个叫做`JSON.parse()`的方法，它将 JSON 格式的数据转换成对象文字格式

```
const JSONdata = '{ "firstName" : "Mike", "lastName" : "Bush"}';
const ObjectData = JSON.parse(JSONdata)

console.log(ObjectData)
```

*输出:-*

```
{firstName: "Mike", lastName: "Bush"}
```

这是我从 JSON 开始学的。理解 JSON 的基本原理和方法很重要，因为它们是使用 API 访问信息的基本部分。

**结论**:

*   JSON 语法类似于 Object literal，其中两个名称-值对都在引号中。
*   `JSON.stringify()`对象>JSON>
*   `JSON.parse()` JSON >对象>

*感谢您花时间阅读本文，请关注我的* [*推特*](https://twitter.com/kadamsarvesh10) *因为我正在记录我的学习*

*原载于*[*https://sarveshcadm . tech*](https://sarveshkadam.tech/getting-started-with-jsonjavascript-object-notation)*。*