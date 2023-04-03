# Python 中的多态性！

> 原文：<https://medium.com/analytics-vidhya/polymorphism-in-python-1fef01f3ec4c?source=collection_archive---------4----------------------->

## *为什么学不会&通过实验探索多态的概念&尝试自己的代码&例子？*

> 在这篇博客中，我想通过**在我的 Jupyter 笔记本上做这些来揭开多态性的神秘面纱，**因为这个概念在不同的网站上有不同的解释&它是用 Python 实现的

**议程:-**
1。)什么是多态性？
2。)多态的内置实现(运算符&函数)
3。)与类方法的多态性
4。)多态性与函数
5。)方法重载
6。)方法覆盖

# 什么是多态性？

多态是面向对象编程的一个概念。

*   多态性这个词意味着有许多形式。
*   在编程中，多态性意味着相同的函数名(但不同的签名)用于不同的类型。
*   如果我们有一个按钮，有许多不同的绘制输出(圆形按钮、复选按钮、方形按钮、带图像的按钮),但是它们共享相同的逻辑:onClick()

> 我们用同样的方法访问它们。这种想法叫做**多态性。**

*   多态性是为不同的底层形式(如数据类型或类)利用相同接口的能力。这允许函数在不同时间使用不同类型的实体。

# **多态性的类型:-**

## 1.多态性的内置实现:-

*   ***多态性中的'+'运算符***

对于整数数据类型，`+`运算符用于执行算术加法运算。

## **多态性加法运算符:-**

```
# Case 1 : When Data Types are Integers
num_1 = 5
num_2 = 10print(num_1+num_2)  #Addition for int Data Types
```

## **输出:-**

```
15
```

类似地，对于字符串数据类型，`+`运算符用于执行连接

```
# Case 2 : When Data Types are Strings
str_1 = "Abhay"
str_2 = "Bansal"print(str_1+" "+str_2) #Concatenation
```

## 输出:-

```
Abhay Bansal
```

这里我们可以注意到，单个操作符`+`被用来根据数据类型执行不同的操作。
这个你可以想到 Python 中最基本的多态。

## 2.多态的内置函数实现:-

len()是 Python 中的一个内置函数，它基于不同的数据类型和结构给出不同的结果或表现出不同的行为。

**len()函数中的多态性:-**

```
print("Length when String :",len("Python"))
print("Length when List:",len(["DataScience","AI","ML"]))
print("Length when Dictionary:",len({"Name":"Abhay","City":"Pune"}))
```

## 输出:-

```
Length when String : 6
Length when List: 3
Length when Dictionary: 2
```

# Python 中的类多态性:-

在直接讨论类方法多态性的概念之前，让我们
举一个例子，问题陈述是计算正方形的面积&矩形的面积。

把自己想象成一名开发人员&让我们想象一下你可以采用什么方法，以及选择哪种方法更好。

![](img/f06801bd7402f2f9a3f4b81c2413ffe2.png)

**方法 1 :-**

*   用不同的名字定义不同的函数:calculate_area_rect()和
    calculate_area_sqr()

```
class Rectangle:
    def calculate_area_rect(self,length,breadth):
        self.length=length
        self.breadth=breadth
        return (self.length*self.breadth)class Square:
    def calculate_area_squar(self,side):
        self.side=side
        return(self.side*self.side)obj_rect = Rectangle()  # Instance of Class Rectangle
obj_squar = Square()    # Instance of Class Squareobj_rect.calculate_area_rect(2,4)
obj_squar.calculate_area_squar(2)print("Area of Rectangle is : ",obj_rect.calculate_area_rect(2,4))print("The Area of Square is : ",obj_squar.calculate_area_squar(4))
```

**输出:-**

```
Area of Rectangle is :  8
Area of Square is :  16
```

这种方法的问题:-
开发者必须记住&计算面积的两个函数名。在一个更大的程序中，很难记住我们执行的每个小操作的函数名。

接下来是**方法重载**的作用，这将在后面详细讨论。

**方法 2:-**

现在，让我们将计算面积的函数的名称改为相同的名称 calculate_area()，同时在两个具有不同定义的类中保持这些函数分开。

```
class Rectangle:
    def calculate_area(self,length,breadth):
        self.length=length
        self.breadth=breadth
        return (self.length*self.breadth)class Square:
    def calculate_area(self,side):
        self.side=side
        return(self.side*self.side)obj_rect = Rectangle()  # Instance of Class Rectangle
obj_squar = Square()    # Instance of Class Squareobj_rect.calculate_area(2,4)
obj_squar.calculate_area(2)print("Area of Rectangle is : ",obj_rect.calculate_area(2,4))print("Area of Square is : ",obj_squar.calculate_area(4))
```

**输出:-**

```
Area of Rectangle is :  8
Area of Square is :  16
```

在这里你可以观察到两个类的实现，例如 Rectangle & Square 有相同的函数名 calculate_area()，但是由于对象不同，它的调用被正确解析。

用 object :
i.)obj_rect 调用 calculate_area()会给出矩形
ii 的面积。)obj_squar 将给出正方形的面积

这种类型的行为是显而易见的，因为我们使用不同的对象，但没有明确证明多态性的定义。

# 类方法的多态性:-

**方法 3:-** 在这种方法中，一切都保持不变，只是我们引入了一个 for 循环，它将对创建的对象元组进行迭代(将不同的对象打包到一个元组中)

```
class Rectangle:
    def __init__(self,length,breadth):
        self.length=length
        self.breadth=breadth

    def cal_area(self):
        return (self.length*self.breadth)class Square:
    def __init__(self,side):
        self.side = side

    def cal_area(self):
        return(self.side*self.side)# Instantiating a Class or creating an Objectobj_rect = Rectangle(2,3)  # Instance of Class Rectangle
obj_squar = Square(2)    # Instance of Class Squarefor obj in(obj_rect,obj_squar):
         print(obj.cal_area())
```

**你可能会想，当我们使用一个循环遍历一组对象时会发生什么？**

**回答:-** 这里 Python 并不关心调用函数的对象的类型，语句只是:obj.cal_area()

现在这是一个更好的多态例子，因为我们把不同类的对象当作一个可以调用相同函数的对象。

您还可以再次回忆一下多态性的定义:

多态支持使用具有不同数据类型输入的单一接口。

# 函数的多态性:-

**方法 4:-** 代替我们在上面的例子中创建的循环，我们也可以创建一个以不同的类对象作为自变量或参数的函数&给出想要的结果。

```
def func(obj):
    return obj.cal_area()# Instantiating a Class or creating an Objectobj_rect = Rectangle(2,3)  # Instance of Class Rectangle
obj_squar = Square(2)    # Instance of Class Squareprint("Area of rectangle is :",func(obj_rect))
print("Area of Square is : ",func(obj_squar))
```

**输出:-**

```
Area of rectangle is : 6
Area of Square is :  4
```

# Python 中的方法重载

**方法 5:-**

> Python 中的两个方法不能有相同的名称(例外情况请参考博客:

[](/analytics-vidhya/unpack-overloading-in-python-da17350c9c75) [## 用 Python 解包重载:-

### 在这篇博客中，我们将按以下顺序详细了解 Python 中的重载:-

medium.com](/analytics-vidhya/unpack-overloading-in-python-da17350c9c75) 

重载是函数或运算符基于传递给函数的参数或运算符所作用的操作数以不同方式表现的能力。

**让我们试着理解下面这段代码:-** 如果 calculate_area()中没有传递任何参数，那么 area 就是 0，如果有一个参数，它假设我们想要的是边*边的正方形的面积，如果传递了两个参数，calculate_area(2，4)就假设这是矩形
的面积，即长度*宽度

```
class FindArea:
    def cal_area(self,length=None,breadth=None):
        self.length=length
        self.breadth=breadth
        if self.length !=None and self.breadth!=None:
            return(self.length*self.breadth)
        elif self.length !=None:
            return(self.length*self.length)
        else:
            return 0obj_area = FindArea()print("Area is :",obj_area.cal_area())
print("Area of Square is :",obj_area.cal_area(4))
print("Area of Rectangle is :",obj_area.cal_area(2,4))
```

**输出:-**

```
Area is : 0
Area of Square is : 16
Area of Rectangle is : 8
```

现在，作为一名开发人员，重新开始思考，试着记住我们之前的例子，我们在 Rectangle & Square 中创建了不同的对象和不同的函数，并使用各自类的对象名来调用它们。

**难道你不认为现在用方法重载开发者是有救了吗？不需要在不同的类中记忆不同的函数名。二。)需要创建单个对象，基于不同的参数，我们可以获得矩形、正方形等的面积。**

# 具有继承性的多态性:-

python 中的多态性定义了子类中与父类中的方法同名的方法。

在继承中，子类从父类继承方法。此外，还可以修改子类中从父类继承的方法。

这主要用于从父类继承的方法不适合子类的情况。这个在子类中重新实现方法的过程被称为**方法覆盖。**

**让我们试着理解下面这段代码:-**

```
class School: # Main Class
    def students_all(self): 
        print("There are many Students in Class")
    def student(self):
        print("This is the student function of Class- School")

class Student_1(School): # Student1 Sub Class of School
    def student(self):
        print( "This is a student function of:Class Student_1")class Student_2(School): # Student2 Sub Class of School
    def student(self):
        print("This is a student function of Class Student_2")

obj_student_2 = Student_2()
obj_student_2.student()obj_student_1 = Student_1()
obj_student_1.student()obj_student = School()
obj_student.student()
```

**输出:-**

```
This is a student function of Class Student_2
This is a student function of:Class Student_1
This is the student function of Class- School
```

由于多态性，Python 解释器自动识别出`student()`方法在类`Student_1` 和类`Student_2` 中被覆盖，并使用在子类中定义的方法。

用 Student_2 `obj_student_2.student()` 的对象调用`student()`方法将首先检查自己类中的`student()`方法，即`Student_2`，如果不存在，它将在父类中寻找`student()`方法，即`School()`

这就是为什么调用`obj_student_2.student()`给出的输出是“这是类 Student_2 的一个学生函数”的原因，与其他函数的方式相同。

不要忘记查看代码库的 GitHub 链接:-[**Python 中的多态性**](https://github.com/bansalabhay/Python-Skill-Set.git)

> **我已经为 Python 中的重载创建了一个详细的博客，其中也考虑到了异常:-**

[](/analytics-vidhya/unpack-overloading-in-python-da17350c9c75) [## 用 Python 解包重载:-

### 在这篇博客中，我们将按以下顺序详细了解 Python 中的重载:-

medium.com](/analytics-vidhya/unpack-overloading-in-python-da17350c9c75) 

## 如果你喜欢这篇文章，请点击👏按钮背书。这将有助于其他媒体用户搜索它。

## **请随时在** [**LinkedIn**](http://www.linkedin.com/in/%20abhay-bansal-0aa374a2) **上联系我，分享你对文章的想法。**

随时欢迎反馈😄