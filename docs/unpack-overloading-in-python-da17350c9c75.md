# 用 Python 解包重载:-

> 原文：<https://medium.com/analytics-vidhya/unpack-overloading-in-python-da17350c9c75?source=collection_archive---------14----------------------->

在这篇博客中，让我们详细了解 Python 中的重载…

![](img/db5cdb7feb5ee94749bd6a1fe0fdf8ac.png)

路易斯·维拉斯米尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

*   *什么是超载？*
*   *优点&缺点*
*   *Python 中的方法重载(以 Java 的方式)及示例*
*   *Python 中的方法重载(自有方式)举例*

**什么是超载？** 重载是函数或运算符基于传递给函数的参数或运算符作用的操作数以不同方式表现的能力。与 Python 相比，Java 或 C++中的重载略有不同。在 Python 中，重载是以不同的方式实现的。

**优点:-**

*   重载一个方法可以实现重用特性。
*   重载也提高了代码的清晰度，消除了复杂性。

**缺点:-**

*   如果过度使用，管理过载的函数会变得很麻烦。

**其他编程语言的方式(如**[**Java**](https://www.javatpoint.com/method-overloading-in-java)**):-** 两个或两个以上的方法或函数可以有相同的名称但参数不同。

基于参数数量调用方法或函数会调用不同的方法或函数。

**让我们用 Python 试试同样的方法:-
例子:-** 创建一个程序，通过方法重载将 2 或 3 个数相加。

```
class Add:
    def add_num(self,a,b):  #first add function
        self.a = a
        self.b = b
        print("Addition of 2 Numbers",a+b)
    def add_num(self,a,b,c): #second add function
        self.a = a
        self.b = b
        self.c = c
        print("Addition of 3 Numbers",a+b+c)

add_obj1 = Add()
add_obj1.add_num(1,2)
```

**输出:-**

```
**TypeError**                                 Traceback (most recent call last)
**<ipython-input-1-e1a2cf687bcf>** in <module>
     14 
     15 add_obj1 **=** Add**()**
**---> 16** add_obj1**.**add_num**(1,2)**

**TypeError**: add_num() missing 1 required positional argument: 'c'
```

用 3 个参数再次调用 add_num()方法:-

```
add_obj1.add_num(1,2,3)
```

**输出:-**

```
Addition of 3 Numbers 6
```

在上面的代码中，我们用 2 & 3 个参数为 add_num()定义了 2 个方法，但是我们只能使用最新的一个。

用 2 个参数调用 add_num()方法会导致参数丢失的错误，而用 3 个参数就可以了。

***现在让我们试着回答下面的问题:-***

***Q1。)Python 中两个方法可以同名吗？*** *答:我们可以在 Python 中重载两个同名的方法，但我们只能使用最新的一个。所以从程序上来说，这是可以做到的，但是没有用。所以我们不应该这样做。*

**Python 之道:-**

**例 1:-** 我们举个例子用 Python 的方式做同样的行为。

```
#Method Overloading for addition of two or three numbersclass Addition:
    def add_num(self,num_1,num_2,num_3=0):
        self.num_1 = num_1
        self.num_2 = num_2
        self.num_3 = num_3
        print("Addition=",self.num_1+self.num_2+self.num_3)obj_add = Addition()obj_add.add_num(2,3) # With Same object and 2 arguments
obj_add.add_num(2,3,4) # With Same object and 3 arguments
```

**输出:-**

```
Addition= 5 # Addition of 2 numbers
Addition= 9 # Addition of 3 numbers
```

在上面的代码中，我们刚刚定义了一个方法 add_num()，并初始化了最后一个数字 num_3=0(因为我认为它是 Int)。这将为我们提供带或不带参数调用它的选项。

当用户使用一个对象调用时&传递 2 个参数，同一个 add_num()给我们加 5，当传递 3 个参数时，它给我们 9。

你不认为我们已经达到了我们的期望吗？

**例 2:**
让我们试着用方法重载来计算矩形或正方形的面积

```
class FindArea:
    def calculate_area(self,length=None,breadth=None):
        self.length=length
        self.breadth=breadth
        if self.length !=None and self.breadth!=None:
            return(self.length*self.breadth)
        elif self.length !=None:
            return(self.length*self.length)
        else:
            return 0obj_area = FindArea()print("Area is :",obj_area.calculate_area())
print("Area of Square is :",obj_area.calculate_area(4))
print("Area of Rectangle is :",obj_area.calculate_area(2,4))
```

**输出:-**

```
Area is : 0
Area of Square is : 16
Area of Rectangle is : 8
```

在上面的例子中，我们将重载 calculate_area()方法，并传递零个、一个和两个参数&使用同一个对象将调用该方法。

如果没有参数，输出为零。
如果有一个参数，这个方法就知道我们在计算正方形的面积。
如果有两个参数，该方法认为我们试图计算矩形的面积，并给出长度*宽度的乘积

要像在 Java 中一样实现重载，我们可以通过安装 decorators 来实现。

**注意:-装饰器功能不在这篇博客中讨论。如果你想让我创建一个详细的，请随时给我发评论。**

不要忘记检查代码库的 GitHub 链接:-[**Python 中的重载**](https://github.com/bansalabhay/Python-Skill-Set.git)

> **如果你想从多态性的基础开始，就点击我的详细博客:-**

[](/analytics-vidhya/polymorphism-in-python-1fef01f3ec4c) [## Python 中的多态性！

### 为什么你不能通过试验和尝试你自己的代码和例子来学习和探索多态性的概念呢？

medium.com](/analytics-vidhya/polymorphism-in-python-1fef01f3ec4c) 

## 如果你喜欢这篇文章，请点击👏按钮背书。这将有助于其他媒体用户搜索它。

## **欢迎随时在**[**LinkedIn**](http://www.linkedin.com/in/%20abhay-bansal-0aa374a2)**上联系我，分享你对文章的想法。**

随时欢迎反馈😄