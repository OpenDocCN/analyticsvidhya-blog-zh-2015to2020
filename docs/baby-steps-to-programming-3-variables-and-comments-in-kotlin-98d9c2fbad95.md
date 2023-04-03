# Kotlin 中编程#3 变量和注释的初级步骤

> 原文：<https://medium.com/analytics-vidhya/baby-steps-to-programming-3-variables-and-comments-in-kotlin-98d9c2fbad95?source=collection_archive---------25----------------------->

![](img/d17f23862422ff7a970144f4530bcf09.png)

照片由[瓦莱里娅·宗科尔](https://unsplash.com/@zoncoll?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/baby-steps-programming?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄

你好，

欢迎回到我们的课程。我们很高兴有你在这里。我希望你过得愉快，当然，我希望编程不再神秘。

今天会很棒，我们会看到变量和注释；编程的一些主要基础知识。

## **变量**

*我给你讲个故事吧……*

电脑是个健忘的牛逼家伙，我是第一次见这个人。我们交谈，我们要点，我们玩*(显然是用它的语言)*我要走了，所以我对它说…嘿！我的名字是西米，请不要忘记，因为我很快就会问你！我的电脑朋友说*“好的，我们很好，我会留着它的”*然后我告诉它*再见👋🏼。*

我离开我的电脑，环游世界，去我从未去过的地方，结识新朋友。然后我回到我的电脑前。

我们互致问候，然后我问它我的名字。我牛逼的电脑没忘记我的名字！它完全按照我告诉它的回忆。我们都很高兴，我们继续成为最好的朋友。

我的电脑故事到此结束。

我希望你喜欢这个故事😀

***电脑是怎么记住我告诉它的话的？***

如此简单，它把它存储在某个地方！*(我们现在不要为那个地方操心了)*

是的，它就是这么做的。此外，它还存储了许多其他东西，所以它在存储的东西上使用了占位符/名称标签，以避免混淆。糟糕的电脑😔。

***用计算机编程术语……***

变量是用来引用计算机存储内容的名称标签或占位符。

看看下面的代码片段。在您的 IDE 中键入它，运行它并尝试理解正在发生的事情。

```
**fun** main(){

    *// tell the computer your name* **var** personsName = **"simi"***// go get yourself breakfast, lunch and dinner here...* *// ask the computer for the name
    println*(**"what is my name?"** +  personsName)

}
```

关键字`var`用来告诉计算机你正在创建一个变量；具有引用名称的存储位置。然后你去照顾你自己，你回来了，在使用变量名`personsName`时，你仍然有你的数据。这是变量的基本知识。

所以，创造你的梦想，尝试新事物。

## **变量类型**

科特林不想让你困惑，所以它只有两种类型的变量

*   `var`:当使用`var`声明一个变量时，它所保存的值可能会改变。
*   `val`:在这种情况下，变量保存的内容不能改变。

看看另一个例子。

```
**fun** main(){

*/*    let me tell you my date of birth
    and my name
    P.S. dont take my word for it.*/* **val** dateOfBirth = **"01/01/1800"
    var** name = **"simi"** *println*(name + **" date of birth is "** + dateOfBirth )

    *// I want to change my name* name = **"Optimus Prime's fan"** *println*(name + **" date of birth is "** + dateOfBirth )

}
```

您可以在 IDE 中键入上面编写的代码，仔细查看并观察运行代码的输出。

***你注意到了什么？***

用`var`声明的变量的值从`“simi”`变成了`“Optimus Prime’s fan”`，计算机没有抱怨，代码运行良好，打印出来的一切都很好。这是一个`var`类型的变量。

***为另一个家伙叫做*** `val`

```
**fun** main(){

*/*    let me tell you my date of birth
    and my name
    P.S. dont take my word for it.*/* **val** dateOfBirth = **"01/01/1800"
    var** name = **"simi"** *println*(name + **" date of birth is "** + dateOfBirth )

    *// I want to change my name* name = **"Optimus Prime's fan"** *println*(name + **" date of birth is "** + dateOfBirth )

    *// I want to lie about my birthdate* dateOfBirth = **"01/01/1999"** *println*(name + **" date of birth is "** + dateOfBirth )
}
```

该代码与我们之前的代码没有太大不同，只是在底部添加了几行代码。试着写并运行这个。*(看！dateOfBirth 是一个* `*val*` *并且我们试图改变它的值)*

你注意到了什么？

![](img/c3b18cb2b2a19e9304ce5c6a40455df3.png)

重新分配变量 val 时出现编译错误

计算机告诉你，a `val`中的值不能改变！因此，您可以将它用于值不会改变的变量。

## 评论

*什么是评论？*

你一定已经注意到在我们写的代码中有一些简单的英语。他们要么跟随`//`要么在`/* ….. */`之间。它们被称为注释。

**意见注释**

*   计算机不会读取/处理它们。
*   所以，你可以用它们来写下你对一堆代码的想法。
*   您可以使用它们来指导您的代码；如果另一个人接触到它，他/她可以知道你的代码中发生了什么。

**评论类型**

Kotlin 编程语言中有两种类型的注释。

*   **单行注释:**它只跨越一行。
    如我们上面的例子。这是一个单行注释:

```
*// I want to change my name*
```

*   **多行注释:**注释可以覆盖多行，电脑全部忽略。取自我们以前的代码，这是一个多行注释。

```
*/*    let me tell you my date of birth
    and my name
    P.S. dont take my word for it.*/*
```

我们今天的工作结束了！

非常感谢你一直坚持到现在。非常感谢。

请继续关注更多关于用 Kotlin 编程的文章。

所有课程的完整源代码可以在这个 repo 中找到:[https://github . com/simi-aluko/baby-steps-to-programming-in-kotlin](https://github.com/simi-aluko/baby-steps-to-programming-in-kotlin)