# JavaScript 中通过值复制任何复合数据类型的 3 种方法

> 原文：<https://medium.com/analytics-vidhya/3-ways-to-copy-by-value-any-composite-data-type-in-javascript-ca3c730e4d2f?source=collection_archive---------7----------------------->

![](img/1ee50cdce9be7fb29e73d6adca952e98.png)

[Fatos Bytyqi](https://unsplash.com/@fatosi?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

1.  使用 spread ( `...`)语法
2.  使用`Object.assign()`方法
3.  使用`JSON.stringify()`和`JSON.parse()`方法

以下说明了如何使用上述三种方法复制对象:

```
const person = {
    firstName: 'John',
    lastName: 'Doe'
}; *// using spread ...*
let p1 = {
    ...person
};*// using  Object.assign() method*
let p2 = Object.assign({}, person);*// using JSON*
let p3 = JSON.parse(JSON.stringify(person));
```

spread ( `...`)和`Object.assign()`都执行浅层复制，而 JSON 方法执行深层复制。

# 浅层拷贝与深层拷贝

在 JavaScript 中，使用[变量](https://www.javascripttutorial.net/javascript-variables/)来存储可以是[原语或引用](https://www.javascripttutorial.net/javascript-primitive-vs-reference-values/)的值。当您复制存储在变量中的值时，您会创建一个具有相同值的新变量。对于原始值，只需使用一个简单的赋值:

```
let counter = 1;
let copiedCounter = counter;
```

当你改变复制变量的值时，原始变量的值保持不变。

```
copiedCounter = 2;
console.log(counter);
```

输出:

```
1
```

但是，如果对引用值使用赋值运算符，它将不会复制该值。相反，两个变量将引用内存中的同一个对象:

```
let person = {
    firstName: 'John',
    lastName: 'Doe'
};
let copiedPerson = person;
```

而当通过新变量(copiedPerson)访问对象并改变它的属性值(name)时，你就改变了对象的属性值。

```
copiedPerson.firstName = 'Jane';
console.log(person);
```

输出:

```
{
    firstName: 'Jane',
    lastName: 'Doe'
}
```

深度复制意味着新变量的值与原始变量断开，而浅复制意味着一些值仍然与原始变量相连。

# 浅层拷贝示例

考虑下面的例子:

```
let person = {
    firstName: 'John',
    lastName: 'Doe',
    address: {
        street: 'North 1st street',
        city: 'San Jose',
        state: 'CA',
        country: 'USA'
    }
}; let copiedPerson = Object.assign({}, person);copiedPerson.firstName = 'Jane'; *// disconnected*copiedPerson.address.street = 'Amphitheatre Parkway'; *// connected*
copiedPerson.address.city = 'Mountain View'; *// connected*console.log(copiedPerson);
```

在本例中:

*   首先，创建一个名为`person`的新对象。
*   其次，使用`Object.assign()`方法克隆`person`对象。
*   第三，更改`copiedPerson`对象的名字和地址信息。

以下是输出:

```
{
    firstName: 'Jane',
    lastName: 'Doe',
    address: {
        street: 'Amphitheatre Parkway',
        city: 'Mountain View',
        state: 'CA',
        country: 'USA'
    }
}
```

但是，当您显示 person 对象的值时，您会发现地址信息发生了变化，但名字却发生了变化:

```
console.log(person);
```

输出:

```
{
    firstName: 'John',
    lastName: 'Doe',
    address: {
        street: 'Amphitheatre Parkway',
        city: 'Mountain View',
        state: 'CA',
        country: 'USA'
    }
}
```

原因是地址是参考值，而名字是原始值。`person`和`copiedPerson`引用不同的对象，但这些对象引用相同的`address`对象。

# 深层拷贝示例

下面的代码片段用 JSON 方法替换了`Object.assign()`方法，以携带`person`对象的深层副本:

```
let person = {
    firstName: 'John',
    lastName: 'Doe',
    address: {
        street: 'North 1st street',
        city: 'San Jose',
        state: 'CA',
        country: 'USA'
    }
}; let copiedPerson = JSON.parse(JSON.stringify(person));copiedPerson.firstName = 'Jane'; *// disconnected*copiedPerson.address.street = 'Amphitheatre Parkway';
copiedPerson.address.city = 'Mountain View';console.log(person);
```

输出

```
{
    firstName: 'John',
    lastName: 'Doe',
    address: {
        street: 'North 1st street',
        city: 'San Jose',
        state: 'CA',
        country: 'USA'
    }
}
```

在这个例子中，`copiedPerson`对象中的所有值都与原始的`person`对象断开。