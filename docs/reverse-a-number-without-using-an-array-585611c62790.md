# 不使用数组反转一个数

> 原文：<https://medium.com/analytics-vidhya/reverse-a-number-without-using-an-array-585611c62790?source=collection_archive---------23----------------------->

![](img/bd11c60bfa01308d00e773fd11c7924d.png)

[https://cartreatments.com/car-wont-go-in-reverse/](https://cartreatments.com/car-wont-go-in-reverse/)

当你在寻找一份新的程序员或软件工程工作时，这是一个常见的面试问题。

当你听到第一部分时，你在天堂。但是当面试官完成最后一部分时，你会崇拜你的数学老师。

当然，它有编程的一面，也有数学的一面。但是当你看到答案的时候，你就再也不用多想了。就这么简单。

```
public static int reverse(int num) { int input = num; int temp = 0; while (input > 0) { temp = temp * 10; temp = temp + input % 10; input = input / 10; } return temp;}
```

这完全是立场的问题。**恭喜**，你通过了一个面试问题！