# Python 函数和函数参数

> 原文：<https://medium.com/analytics-vidhya/python-function-and-function-parameters-3d4d7f13b7fe?source=collection_archive---------19----------------------->

![](img/453b404c790dc27d2b7500397692f9a2.png)

知识是没有海岸的海洋

简而言之，当我们定义一个函数时，我们可能会也可能不会传递参数。但是大多数时间参数是在定义函数时传递的。语义上，我们将函数定义如下:

> def myfunction(a，b):
> #一些代码

所以当我们调用这个函数时，

> x = 10
> y = 20
> 我的函数(x，y)

在这个上下文中， ***x*** 和 ***y*** 是 ***myfunction*** 的参数，最重要的是要注意， ***x*** 和 ***y*** 是通过引用传递的。即 ***x*** 和 ***y*** 的内存地址通过。
我们可以将*位置*和*关键字*参数传递给函数。

## **位置和关键字参数**

我们通常按照参数传递的顺序给函数的参数赋值，也就是参数的位置。

让我们定义一个函数，它接受两个参数并返回两个字符串的连接。

```
def concatenate(a: str, b: str) -> str:
    return a + b
```

所以当我们通过传递参数调用上述函数时，

```
concatenate(**"hello"**, **"world!!"**)
```

其中第一个参数" hello "赋给第一个参数即 a 和第二个参数" **world！！**"分配给第二个参数，即 b

我们可以通过为相应的参数指定一个“缺省”值来使位置变量成为可选的。让我们通过添加额外参数来修改我们的*连接*函数，

```
def concatenate(a: str, b: str, c: str = **"Welcome"**) -> str:
    return a + b + c
```

我们添加了第三个变量*“c”*，默认值为“*欢迎使用*”。因此，如果我们调用上述函数而不传递第三个参数，那么将引用第三个位置参数默认值。

所以如果我们调用上面的函数，

```
print(concatenate(**"hello"**, **" world!! "**))
```

输出将是:

> *你好世界！！欢迎光临*

现在，如果我们传递第三个参数，那么函数将不会考虑第三个参数的默认值(这是显而易见的😊).为了举例说明，让我们修改我们的代码:

```
print(concatenate(**"Hello, "**,**" this is my first story in "**, **" Medium"**))
```

当我们运行上面的代码时，输出将是:

> *你好，这是我在 Medium 的第一个故事*

到目前为止，一切都很好，也很容易理解。然而，可能会有这样的情况，函数中的任何一个参数都是可选的。更准确地说，假设我们有三个参数的加法函数，其中一个或者简化为第二个位置参数是可选的。

```
def addition(a: int, b: int = 150, c: int) -> int:
    return a + b + c
```

如果我们只传递两个参数来调用这个函数，例如:

```
print(addition(150,150))
```

那么 python 将无法识别第二个参数即 *150* 是指第二个参数即 *b* 还是第三个参数即 *c* 。所以它会在执行时抛出错误。

错误可能是这样的:

```
**def addition(a: int, b: int = 150, c: int) -> int:
             ^**
   ** SyntaxError: non - default argument follows default argument**
```

因此，为了避免这种错误，我们需要在用默认值传递位置参数时遵循以下规则。

> **如果我们用默认值定义了一个位置参数，那么对于其后的每个位置参数，我们也必须被赋予默认值。**

所以我们需要修改我们的*加法*函数，将默认值赋给第三个参数，即**c**

```
def addition(a: int, b: int = 150, c: int = 0) -> int:
    return a + b + c
```

现在当我们运行下面的代码时:

```
print(addition(200, 300)) # a = 200, b = 300, c = 0
```

输出将是:

> 500

对于加法函数，我们现在也可以传递一个参数。例如:

```
print(addition(1)) # a = 1, b = 150 (Default), c = 0(Default)
```

输出将是:

> 151

现在，如果你想跳过第二个参数，将第一个和第三个参数传递给加法函数。我们可以通过使用“ ***关键字参数*** ”来实现，这也称为“ ***命名参数*** ”。

使用关键字或命名参数，我们可以通过传递第一个和第三个参数来调用函数，第二个参数保留默认值。

```
print(addition(a=1000, c=1500)) # So b = 150 (default value)
```

输出将是:

> 2650

**备注:**

> 1.我们可以通过使用参数名来指定位置变量，不管它们是否赋值。

示例:让我们定义一个带有 4 个参数的函数，并返回传递给该函数的所有 4 个值。

```
def returning_value(val1, val2, val3, val4):
    return **"val1 = {0}, val2 ={1}, val3 ={2}, val4 ={3}"**.format(val1, val2, val3, val4)
```

您可以调用函数，或者不使用关键字参数传递所有四个参数，或者使用关键字参数。

```
*# Passing arguments without keywords argument* print(returning_value(1, 2, 3, 4))
```

输出将是:

> 值 1 = 1，值 2 =2，值 3 =3，值 4 =4

```
*# Passing arguments with keywords argument* print(returning_value(val4=400, val1=120, val3=300, val2=140))
```

输出将是:

> val1 = 120，val2 =140，val3 =300，val4 =400

如果我们使用关键字参数，那么传递参数的顺序并不重要，这就是使用关键字参数的优点。

然而，使用关键字参数时有一个警告。

> 2.如果您开始使用关键字参数，那么其后的所有参数都必须是关键字参数。

例如:

```
*# Passing few argument with keyword* print(returning_value(val1=100, 200, val3=300, 400))
```

当我们运行上面的代码时，会显示下面的错误信息:

```
**print(returning_value(val1=100, 200, val3=300, 400))
                                   ^
SyntaxError: positional argument follows keyword argument**
```

## **任意参数(*args)**

当我们定义一个带参数的函数并访问带参数的函数时，最终我们是通过参数的相对位置来访问参数的。

例如， *my_returnargs* 函数接受 3 个位置参数并返回这些值。

```
**def** my_returnargs(val1, val2, val3):
    print(**"First Positional Value = "**, val1)
    print(**"Second Positional Value = "**, val2)
    print(**"Third Positional Value = "**, val3)
```

让我们定义一个列表变量，它有三个值:

```
list = [**'Hello'**, **'World'**, **'Learn Python'**]
```

现在让我们将 list 作为一个参数传递给 *my_returnargs* 函数，我们将解包 list，这样所有的值都将作为一个参数传递:

```
my_returnargs(*list)
```

输出将是

> 第一位置值= Hello
> 第二位置值= World
> 第三位置值= Learn Python

在上面的例子中，我们在函数中传递了 3 个参数，因为 list 有 3 个元素，所以在解包时，所有三个参数都被赋值。

现在，如果我们有一个超过 3 个元素的相同列表，并试图在函数参数中解包，会发生什么？

```
list = [**'Hello'**, **'World'**, **'Learn Python'**, **'Some Extra Arguments'**]

my_returnargs(*list)
```

所以当我们执行上面的代码时，python 会抛出错误:

```
my_returnargs(*list)
**TypeError: my_returnargs() takes 3 positional arguments but 4 were given**
```

我们可以通过在函数中传递任意参数(*args)来避免这种错误。我们可以修改这个函数，接受两个位置参数，并传递任意参数。

```
**def** my_returnargs(val1, val2, *args):
    print(**"First Positional Value = "**, val1)
    print(**"Second Positional Value = "**, val2)
    print(**"Arbitrary Argument Values = "**, args)
```

现在让我们运行同样的代码，在这里我们已经用 4 个值解包了这个列表

```
list = [**'Hello'**, **'World'**, **'Learn Python'**, **'Some Extra Arguments'**]

my_returnargs(*list)
```

输出将是:

> 第一个位置值= Hello
> 第二个位置值= World
> 任意参数值=('学习 Python '，'一些额外的参数')

**注意**:args 参数返回值为**元组**

> 也不能在*args 后面添加任何位置参数。这意味着*args 最终会用尽所有的位置参数。

我们可以通过在*args 后传递位置参数来定义函数。Python 在定义这样的函数时绝不会抱怨:)

```
**def** my_returnargs(val1, val2, *args, val3):
    print(**"First Positional Value = "**, val1)
    print(**"Second Positional Value = "**, val2)
    print(**"Arbitrary Argument Values = "**, args)
    print(**"Another positional argument Value = "**, val3)
```

当我们通过传递参数调用函数时会出现问题，python 会抛出 TypeError

```
list = [**'Hello'**, **'World'**, **'Learn Python'**, **'Some Extra Arguments'**]

my_returnargs(*list)
```

输出:

```
my_returnargs(*list)
TypeError: my_returnargs() missing 1 required keyword-only argument: **'val3'**
```

然而，我们可以通过强制用户传递强制关键字参数来解决此类问题。

## 关键字参数*

什么时候我们可以使用强制的关键字参数？所以答案是，一旦我们用尽了所有的位置论点。

为了说明，让我们看下面的例子:

```
**def** my_returnargs(val1, val2, *args, val3):
    print(**"First Positional Value = "**, val1)
    print(**"Second Positional Value = "**, val2)
    print(**"Arbitrary Argument Values = "**, args)
    print(**"Another positional argument Value = "**, val3)

list = [**'Hello'**, **'World'**, **'Learn Python'**, **'Some Extra Arguments'**]

my_returnargs(*list, val3 = **"This is a keyword argument"**) *# Explicitly pass the keyword argument*
```

当我们运行上面的代码时，输出将是:

> 第一个位置值= Hello
> 第二个位置值= World
> 任意参数值=('学习 Python '，'一些额外的参数')
> 另一个位置参数值=这是一个关键字参数

我们可以通过传递*作为参数来显式地限制函数不传递任何位置参数。例如:

```
**def** no_positionalarg(*, val: str) -> str:
    print(**"Only Keyword Argument :"**, val)
```

如果我们试图将位置参数和关键字参数一起传递，那么 python 将通过 TypeError

```
no_positionalarg(10,20,val = **"Hello World!!"**)
```

输出将是:

```
no_positionalarg(10,20,val = **"Hello World!!"**)
**TypeError: no_positionalarg() takes 0 positional arguments but 2 positional arguments (and 1 keyword-only argument) were given**
```

但是如果我们只传递关键字参数，那么 Python 不会显示任何错误

```
no_positionalarg(val = **"Hello World!!"**)
```

输出将是:

> 唯一关键字参数:Hello World！！

现在让我们结合位置参数，可选的位置参数，*args，无位置参数和强制关键字参数。

```
**def** combine_func(val1, val2=50, *args, val3, val4=**"Hello World!!!"**):
    print(**"val1 = "**, val1)
    print(**"val2 = "**, val2)
    print(**"args = "**, args)
    print(**"val3 = "**, val3)
    print(**"val4 = "**, val4)
```

在上面的函数中，我们有:

> 强制位置参数，即 *val1*
> 
> 可选位置参数，即 *val2*
> 
> 可选任意数量的位置参数，即*args
> 
> 强制关键字参数，即 *val3*
> 
> 最后可选位置参数，即 *val4*

所以当我们调用这个函数时:

```
list = [101, **'Python'**, **'Hello'**, **'Welcome'**, 20, 30]
combine_func(*list, val3=**"This is a mandatory keyword arguments after args"**)
```

输出将是:

> val 1 = 101
> val 2 = Python
> args =(' Hello '，' Welcome '，20，30)
> val3 =这是 args
> val4 = Hello World 之后的强制关键字参数！！！

如果我们在函数中不限制位置参数，然后是强制关键字参数和可选位置参数，会怎么样？

让我们修改如下相同的函数:

```
**def** combine_func(val1, val2=50, *, val3, val4=**"Hello World!!!"**):
    print(**"val1 = "**, val1)
    print(**"val2 = "**, val2)
    print(**"val3 = "**, val3)
    print(**"val4 = "**, val4)
```

在上面的函数中，我们有:

> 强制位置参数，即 *val1*
> 
> 可选位置参数，即 *val2*
> 
> 没有位置参数，即 ***
> 
> 强制关键字参数，即 *val3*
> 
> 最后可选位置参数，即 *val4*

所以当我们通过传递三个位置参数后跟关键字参数来调用函数时

```
combine_func(**'test'**,25,1000,c=100.1005)
```

然后我们会遇到下面的错误:

```
combine_func(**'test'**,25,1000,val3=100.1005)
**TypeError: combine_func() takes from 1 to 2 positional arguments but 3 positional arguments (and 1 keyword-only argument) were given**
```

原因是前两个参数指的是位置参数，即 ***val1*** 和 ***val2*** (虽然 ***val2*** 是可选的位置参数，其默认值为 *50* ，但这里我们传递的是 *25* )。) .然而，当我们在关键字参数之前传递第三个参数，即 *1000* ，即 ***val3*** 时，Python 无法处理，因为函数被定义为在两个位置参数之后，不应传递任何位置参数，而是传递强制关键字参数，即 ***val3*** ，后跟另一个可选位置参数。

让我们在调用函数时修改参数，如下所示:

```
combine_func(**'test'**,25,val3=100.1005, val4 = **"Welcome to pythonic way of writing python"**)
```

输出将是:

> val 1 = test
> val 2 = 25
> val 3 = 100.1005
> val 4 =欢迎来到 python 的编写方式 python

**非常有趣但非常重要(更确切地说，这是一个警告)，如果你调用一个带有命名参数或关键字参数的函数，尽管你声明它们是位置参数后跟*args，那么 Python 将抛出语法错误，声明位置参数跟在关键字参数后面。**

为了说明这一点，让我们运行下面的代码:

```
**def** namedargument_with_args(val1, val2, val3=10, *args):
    print(val1, val2, val3, *args)

namedargument_with_args(val1=1, val2=2, 30, 50, 60, 80)
```

所以当我们运行上面的代码时，会显示下面的错误:

```
namedargument_with_args(val1=1, val2=2, 30, 50, 60, 80)
                                           ^
**SyntaxError: positional argument follows keyword argument**
```

如果我们调用函数并试图在*args 的末尾传递 option 参数会怎样？当然 Python 会抛出错误，因为它发现了位置参数 *val3* 的多个值。

```
**def** namedargument_with_args(val1, val2, val3=10, *args):
    print(val1, val2, val3, *args)

namedargument_with_args(1,2, 30, 50,60,80, val3= 20)**>>> Output:**
namedargument_with_args(1,2, 30, 50,60,80, val3= 20)
**TypeError: namedargument_with_args() got multiple values for argument 'val3'**
```

## 任意数量的关键字参数(**kwargs)

在 python 中，我们可以使用**kwargs 向函数添加任意数量的关键字参数。

即使位置参数没有用尽，用户也可以指定**kwargs(这对于*args 是不成立的)。但是，**kwargs 后面不应该有任何位置参数。

但是有一个警告。如果我们声明一个第一个参数是*的函数，也就是说，后面没有位置参数，那么 Python 不会处理，而是抛出一个语法错误:

```
**def** kwargs_function(*,**kwargs):
    print (kwargs)**>>> Output:**
    def my_func(*,**kwargs):
                 ^
**SyntaxError: named arguments must follow bare ***
```

但是，如果我们在**kwargs 后传递强制关键字或可选位置参数，上面的函数将会工作。

```
**def** kwargs_function(*,val,**kwargs):
    print (**"The keyword argument value = "**, val)
    print (**"Arbitrary Keyword Arguments are: "**, kwargs)

kwargs_function(val= **"Hello World!"**, val1 = 1, val2 = 2, val3 = 3, val4 = 4, val5 = 5, val6 = **"Welcome to Medium!!!!"**)**>>>** **Output**:
The keyword argument value =  **Hello World!**
Arbitrary Keyword Arguments are:  **{'val1': 1, 'val2': 2, 'val3': 3, 'val4': 4, 'val5': 5, 'val6': 'Welcome to Medium!!!!'}**
```

**注意**:kwargs 参数返回值为**字典**

我们可以在一个函数中同时使用*args 和**kwargs。

```
**def** args_kwargs_func(*args,**kwargs):
    print(**"Arbitrary arguments values :"**, args)
    print (**"Arbitrary Keyword Arguments are: "**, kwargs)

args_kwargs_func(10, 20, 30 , 40 ,50 ,a = 1, b = 2, c = 3, d = 4, e = 5, f = **"Hello World!!"** )**>>> Output** Arbitrary arguments values : (10, 20, 30, 40, 50)
Arbitrary Keyword Arguments are:  {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 'Hello World!!'}
```

我们可以用位置参数、任意数量的位置参数(即*args)、强制的仅关键字参数和任意关键字参数来声明函数。

```
**def** my_func(arg_1, arg_2, *args, kwarg_1, **kwargs):
    print(arg_1)
    print(arg_2)
    print(args)
    print(kwarg_1)
    print(kwargs)

s1 = **"Welcome"** s2 = **"Python"** tuple_1 = (1, 2, 3, 4, 5, 6, **"Hello World!!!"**)
my_func(s1, s2, *tuple_1, kwarg_1=**"Mandatory Keyword Argument"**, a=10, b=20, c=**"Hello"**, d=**"World!"**)**>>> Output** Welcome
Python
(1, 2, 3, 4, 5, 6, 'Hello World!!!')
Mandatory Keyword Argument
{'a': 10, 'b': 20, 'c': 'Hello', 'd': 'World!'}
```

在定义函数时，我们可以传递位置参数、关键字参数以及任意参数&任意关键字参数。然而，我们需要注意传递这些参数的方式，即在定义任何函数时要记住某些规则。

也请发表您的疑问/想法和建议:)

快乐学习！！！