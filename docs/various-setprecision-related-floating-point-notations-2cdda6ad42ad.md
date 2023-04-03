# 各种与 precision()相关的浮点符号

> 原文：<https://medium.com/analytics-vidhya/various-setprecision-related-floating-point-notations-2cdda6ad42ad?source=collection_archive---------27----------------------->

在标准 C++中，格式化是在某些变量、操纵器、对象等的帮助下完成的。驻留在 ostream 默认类的输出流(cout)中。主要只有两种类型的浮点操作器，著名的固定浮点记数法和科学浮点记数法，后者以与“e”的关系而闻名(解释如下)。这些都在广泛使用的<iostream>头文件库中定义。</iostream>

**为什么设置 precision()？**

precision()在标准浮点表示法中的基本用途是指定最大数

必须显示有效数字，包括数字的所有位数，无论它出现在小数点之前还是之后，而在固定记数法和科学记数法中，精度函数指定小数点之后要显示的确切位数。

现在让我们理解所有可以和 precision()函数一起使用的浮点符号。它们如下:

## 1.浮点符号(标准::固定) :

在这种情况下，我们用固定的符号书写浮点值。最终值显示在小数部分之后的精确数字，小数部分作为精度(x)中的参数给出，这里是 x。在这种情况下，我们不计算指数部分。

代码:`#include <iostream>`

`**using**` `**namespace**` `std;`

`**int**` `main()`

`{`

`// Initializing`

`**double**`

`**double**`

`//Giving precision parameter`

`cout.precision(4);`

`// Printing normal values`

`cout << "Normal values of floating point numbers\na = ";`

`cout << a << "\nb = "` `<< b << '\n'` `;`

`// Printing values using fixed ( till 4 )`

`cout << "Values using fixed \n"` `<< std::fixed;`

`cout << a << "\n"` `<< b << '\n'` `;`

`**return**` `0;`

`}`

输出:

浮点数的正常值
a = 4.223
b = 2323
使用固定值
4.2232
2323.0000

## 2.科学浮点表示法(std::scientific):

在这个例子中，它用科学记数法显示浮点值。类似于任何科学值，在小数部分之前只有一位数字，在小数点之后的位数与在 setprecision(x)中作为参数给出的数字相同，这里是 x。这个符号有一个特殊的部分(正如我上面提到的)，它总是包括一个 e，这是一个指数部分，后面跟着一个可选的符号和三个指数数字。

代码:

`#include <iostream>`

`**using**` `**namespace**` `std;`

`**int**` `main()`

`{`

`// Initializing floating point variable`

`**double**`

`**double**` `b = 2323.0;`

`// Specifying precision`

`cout.precision(4);`

`// Printing normal values`

`cout << "Normal values of floating point numbers\na = ";`

`cout << a << "\nb = "` `<< b << '\n'` `;`

`// Printing values using scientific ( till 4 )`

`// after 4, exponent is used`

`cout << "Values using scientific are : "`

`cout << a << '\n'` `<< b << '\n'` `;`

`**return**` `0;`

`}`

输出:

```
Normal values of floating point numbers
a = 4.223
b = 2323
Values using scientific are : 
4.2232e+00
2.3230e+03
```

## 3.Hexfloat 浮点表示法(std::hexfloat):

这个函数在精度参数(x)被传递后，显示转换成十六进制形式后的期望数字。

代码；

`#include <iostream>`

`**using**``**namespace**`

`**int**` `main()`

`{`

`// Initializing floating point variable`

`**double**` `a = 4.223234232;`

`**double**` `b = 2323.0;`

`// Specifying precision`

`cout.precision(4);`

`// Printing normal values`

`cout << "Normal values of floating point numbers\na = ";`

`cout << a << "\nb = "` `<< b << '\n'` `;`

`// Printing values using hexfloat ( till 4 )`

`cout << "Values using hexfloat are : "` `<< std::hexfloat << endl;`

`cout << a << '\n'` `<< b << '\n'` `;`

`**return**`

`}`

输出:

```
Normal values of floating point numbers
a = 4.223
b = 2323
Values using hexfloat are : 
0x1.0e49783b72695p+2
0x1.226p+11
```

## 4.**默认浮点浮点表示法(std :: default float )** :

定义精度(x)后，它显示所需的数字，与默认值相同。这个主要用于区分其他使用的格式，以便更好地判断代码。

代码:

`#include <iostream>`

`**using**` `**namespace**` `std;`

`**int**` `main()`

`{`

`// Initializing floating point variable`

`**double**` `a = 4.223234232;`

`**double**` `b = 2323.0;`

`// Specifying precision`

`cout.precision(4);`

`// Printing normal values`

`cout << "Normal values of floating point numbers\na = ";`

`cout << a << "\nb = "` `<< b << '\n'` `;`

`// Printing values using defaultfloat ( till 4 )`

`// same as normal`

`cout << "Values using defaultfloat are : "` `<< std::defaultfloat << endl;`

`cout << a << '\n'` `<< b << '\n'` `;`

`**return**`

`}`

输出

```
Values using fixed 
4.2232
2323.0000
Values using defaultfloat are : 
4.223
2323
```

希望您已经理解了精确使用的浮点表示法的基本定义，并且现在将更加自信地使用它们。

![](img/5c9e05f46a105c095a999ccae0496d38.png)