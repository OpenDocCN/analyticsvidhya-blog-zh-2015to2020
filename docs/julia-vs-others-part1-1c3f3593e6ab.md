# 朱莉娅和其他人-第一部分

> 原文：<https://medium.com/analytics-vidhya/julia-vs-others-part1-1c3f3593e6ab?source=collection_archive---------24----------------------->

Julia 是一种科学的编程语言。它与 Python 和 C 特别相似，它的结构有些类似于 C，有些类似于 Python。

![](img/b8beac3ea8c8db6d1c94744f7b1f15e5.png)

朱莉娅跑起来比 Python 快；这种编码结构比 C 和 Python 更简单。它有像 C 一样的结构对象，但没有像 Python 一样的类对象。然而，Julia 可以像面向对象的语言一样重载函数。一个函数可以为不同类型对象重新定义一个以上。Julia 的数组从 1 开始，而不是从 0 开始。

一般来说，我会比较 Julia 和 Python 代码编写风格。首先，我将从控制流中的循环开始。在 Julia 中，控制流以“end”语句结束。而且，一些科学符号可以用在代码块中，比如∈(集合中的元素，\in)。如果你想使用一些符号，你可以在 [Unicode 输入](https://docs.julialang.org/en/v1/manual/unicode-input/)上找到它们。

## Python For 循环

```
**Exp 1 :**prog_lang = "Python"for i in range(len(prog_lang)):

    print(i, ". element : ", prog_lang[i])**Exp 2 :**elements = ["a", "b", "c"]for element in elements:

    print(element)
```

## 朱莉娅为循环

```
**Exp 1.a :**prog_lang = "Julia"for i in ***1***:length(prog_lang)

    println(i, ". element : ", prog_lang[i])end**Exp 1.b :**prog_lang = "Julia"for i ***∈*** ***1***:length(prog_lang)

    println(i, ". element : ", prog_lang[i])end**Exp 2.a :**elements = ["a", "b", "c"]for element in elementsprintln(element)

end**Exp 2.b :**elements = ["a", "b", "c"]for element ***∈*** elementsprintln(element)

end
```

另一种不可或缺的循环类型是 while。julia 有“println”功能，可以在屏幕上显示一些消息，比如 Python 的 print 命令。朱莉娅没有“白”字结构。但是，可以使用“中断”命令来管理这种情况。

## Python While 循环

```
**Exp 1 :**input_value = 20mod_number = 2while True:

    if (mod_number > (input_value / 2)):

        break

    if ((input_value % mod_number) == 0):

        print(mod_number)

    mod_number += 1**Exp 2 :**input_value = 11numbers = 0while input_value <= 10:

    print(numbers)

    numbers += 1

else:

    print(input_value, " is not equal or less than 10...")
```

## Julia While 循环

```
**Exp 1 :**input_value = 20mod_number = 2while ***true***

    if (mod_number > (input_value / 2))

        break

    end

    if ((input_value % mod_number) == 0)

        println(mod_number)

    end

    mod_number += 1end**Exp 2 :**input_value = 11

numbers = 0

while ***true***

    ***if (input_value > 10)

      println(input_value, " is not equal or less than 10...")
      break

    end***

    println(numbers)

    numbers += 1

end
```

If-Else 控制流对于一个程序来说非常重要。这就是为什么我想回到这个结构，即使我已经给出了一些例子。

## Python If/Else 流

```
**Exp 1 :**num_value = 6if num_value == 2:

    print("This number is 2\. And, it is control threshold...")elif num_value > 2 :

    if (num_value % 2) == 0:

        print("This number is even...")

    else:

        print("This number is odd...")

else:

    print("Please, enter number is bigger than 2...")
```

## 朱莉娅 If/Else 流程

```
**Exp 1 :**num_value = 6

if num_value == 2

    println("This number is 2\. And, it is control threshold...")

***elseif*** num_value > 2

    if (num_value % 2) == 0

        println("This number is even...")

    else

        println("This number is odd...")

    end

else

    ***print***("Please, enter number is bigger than 2...")

end
```

有时，我们希望在一行中编写基本的 if 子句…

**巨蟒一行**

```
**Ex 1 :**x = 10
y = 5response = “Equals” if x == y else “Not Equal”print(response)
```

**朱丽亚一线**

```
**Ex 1 :**x = 10
y = 5response = x == y ***?*** “Equals” ***:*** “Not Equal”***print***(response)
```

> 然后
> 
> 复合表达式，函数定义和结构对象将在第二部分…