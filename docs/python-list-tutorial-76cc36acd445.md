# Python 列表教程

> 原文：<https://medium.com/analytics-vidhya/python-list-tutorial-76cc36acd445?source=collection_archive---------15----------------------->

![](img/18e4d54910708d82cefaa066d0440c9f.png)

# 部分概述

*   列表
*   部分
*   异常处理
*   遍历列表
*   排序和范围
*   范围

# 第一节。列表

*   列表是 Python 中最强大的数据类型之一。
*   它是一种保存有序项目集合的数据类型。
*   这些项目可以是各种数据类型。你甚至可以有列表的列表。

列表是使用方括号之间的逗号分隔值创建的。`The format is:`

```
list_name = [item_1, item_2, ..., item_N]
list_name = []
```

利用我们对 Python 类型的理解，我们可能认为可以将每个数据点存储在它自己的变量中。例如，我们可以这样存储第一行的数据点:

```
first_name = 'Erdi'
last_name = 'Mollahüseyin'
email = 'erdimollahuseyin@gmail.com'
age = 29
```

上面，我们存储了:

*   第一个名字“Erdi”为字符串
*   姓氏“Mollahüseyin”为字符串
*   电子邮件"[erdimollahuseyin@gmail.com](mailto:erdimollahuseyin@gmail.com)"作为字符串
*   作为整数的 29 岁

我们可以使用列表更有效地存储数据。这就是我们如何为第一行创建数据点列表:

```
user_row_1 = ['Erdi', 'Mollahüseyin', 'erdimollahuseyin@gmail.com', 29]print(user_row_1)
['Erdi', 'Mollahüseyin', 'erdimollahuseyin@gmail.com', 29]type(user_row_1)
list
```

创建列表后，我们通过将它赋给一个名为 users 的变量，将其存储在计算机的内存中。

要创建数据点列表，我们只需:

*   用逗号分隔数据点。
*   用括号将数据点序列括起来。

现在让我们创建 3 个列表:

```
user_row_1 = ['Erdi', 'Mollahüseyin', 'erdimollahuseyin@gmail.com', 29]
user_row_2 = ['Cengiz', 'Under', 'cengizunder@me.com', 23]
user_row_3 = ['Burak', 'Yilmaz', 'burakyilmaz@me.com', 32]
```

*   像[7，8，9]这样的列表有相同的数据类型(只有整数)
*   而列表['Erdi '，' Mollahüseyin '，'[erdimollahuseyin@gmail.com](mailto:erdimollahuseyin@gmail.com)'，29]具有混合数据类型:
*   三弦('尔迪'，'莫拉赫色音'，'[erdimollahuseyin@gmail.com](mailto:erdimollahuseyin@gmail.com)')
*   一个整数(29)

['Erdi '，' Mollahüseyin '，'[erdimollahuseyin@gmail.com](mailto:erdimollahuseyin@gmail.com)'，29]列表有四个数据点。要找到一个列表的长度，我们可以使用`len()`命令:

```
user_row_1 = ['Erdi', 'Mollahüseyin', 'erdimollahuseyin@gmail.com', 29]
print(len(user_row_1))
4list_1 = [7, 8, 9] 
print(len(list_1))
3list_2 = []
print(len(list_2))
0
```

另一个例子是饮料列表，列表中的项目可以通过索引来访问。列表索引是从零开始的。`The format is:`

```
list_name[index]drinks = ['water', 'tea', 'orange juice', 'beer']
print(drinks)
['water', 'tea', 'orange juice', 'beer']type(drinks)
listprint(drinks[0])
waterprint(drinks[1])
teaprint(drinks[2])
orange juiceprint(drinks[3])
beerdrinks[0] = 'cider'
print(drinks[0])
cider
```

使用负索引从列表的末尾访问项目。`The last item in a list is:`

```
list_name[-1]drinks = ['water', 'tea', 'orange juice', 'beer']print(drinks[-1])
beerprint(drinks[-2])
orange juiceprint(drinks[-3])
teaprint(drinks[-3])
water
```

使用`append()`或`extend()`列表方法向列表添加项目。

```
drinks = ['water', 'tea', 'orange juice', 'beer']drinks.append('wine')
print(drinks[-1])
winedrinks = ['water', 'tea', 'orange juice', 'beer']drinks.extend(['cider', 'wine'])
print(drinks)
['water', 'tea', 'orange juice', 'beer', 'cider', 'wine']drinks = ['water', 'tea', 'orange juice', 'beer']
more_drinks = ['cider', 'wine']
drinks.extend(more_drinks)
print(drinks)
['water', 'tea', 'orange juice', 'beer', 'cider', 'wine']drinks = ['water', 'tea', 'orange juice', 'beer']
drinks.insert(0, 'wine')
print(drinks)
['wine', 'water', 'tea', 'orange juice', 'beer']drinks.insert(2, 'cider')
print(drinks)
['wine', 'water', 'cider', 'tea', 'orange juice', 'beer']
```

# 第二节。部分

使用切片访问列表的一部分。`The format is:`

```
list_name(start, stop)
list[index1:index2]
list[:index2]
list[index1:]drinks = ['wine', 'water', 'cider', 'tea', 'orange juice', 'beer']some_drinks = drinks[1:4]
print('Some drinks: {}'.format(some_drinks))
Some drinks: ['water', 'cider', 'tea']first_two = drinks[0:2]
print('First two drinks: {}'.format(first_two))
First two drinks: ['wine', 'water']first_two_again = drinks[:2]
print('First two drinks: {}'.format(first_two_again))
First two drinks: ['wine', 'water']drinks = ['wine', 'water', 'cider', 'tea', 'orange juice', 'beer']
last_two = drinks[4:6]
print('Last two drinks: {}'.format(last_two))
Last two drinks: ['orange juice', 'beer']last_two_again = drinks[-2:]
print('Last two drinks: {}'.format(last_two_again))
Last two drinks: ['orange juice', 'beer'] # String Slices
part_of_a_water = 'water'[1:3]
print(part_of_a_water)
at
```

# 第三节。异常处理

list `index()`方法接受一个值作为参数，并返回列表中第一个值的索引，如果该值不在列表中，则返回异常。

```
# Finding an item in a list.
drinks = ['wine', 'water', 'cider', 'tea', 'orange juice', 'beer']
drinks_index = drinks.index('water')
print(drinks_index)
1# Exception
drinks = ['water', 'tea', 'orange juice', 'beer']
wine_index = drinks.index('wine')
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-42-d14f756f0077> in <module>
----> 1 wine_index = drinks.index('wine')ValueError: 'wine' is not in listdrinks = ['water', 'tea', 'orange juice', 'beer']
try: 
    wine_index = drinks.index('wine') 
except: 
    wine_index = 'No wines found.' 
print(wine_index)
No wines found.
```

未处理的异常导致 Python 程序终止。使用`try/except`模块处理异常。

# 第四节。遍历列表

## 第 4.1 节 FOR —循环

使用 for 循环遍历列表。`The format is:`

```
for item_variable in list_name:
    # Code blockitem_variable = list[0]
item_variable = list[1]
item_variable = list[N]drinks = ['water', 'tea', 'orange juice', 'beer']
for drink in drinks:
    print(drink.upper())
WATER
TEA
ORANGE JUICE
BEER
```

## 第 4.2 节 WHILE —循环

只要条件的计算结果为 true，while 循环中的代码块就会执行。`The format is:`

```
while condition
    # Code blockdrinks = ['water', 'tea', 'orange juice', 'beer']
index = 0while index < len(drinks):
    print(drinks[index])
    index += 1
water
tea
orange juice
beer
```

# 第五节。排序和范围

要对列表进行排序，请使用`sort()`列表方法或 hee 内置的`sorted()`函数。

```
drinks = ['water', 'tea', 'orange juice', 'beer']
print('Drinks list: {}'.format(drinks))
Drinks list: ['water', 'tea', 'orange juice', 'beer']sorted_drinks = sorted(drinks)
print('Sorted animals list: {}'.format(sorted_drinks))
Sorted animals list: ['beer', 'orange juice', 'tea', 'water']drinks.sort()
print('Drinks after sort method: {}'.format(drinks))
Drinks after sort method: ['beer', 'orange juice', 'tea', 'water']drinks = ['water', 'tea', 'orange juice', 'beer']
more_drinks = ['wine']
all_drinks = drinks + more_drinks
print(all_drinks)
['water', 'tea', 'orange juice', 'beer', 'wine']drinks = ['water', 'tea', 'orange juice', 'beer']
print(len(drinks))
4drinks.append('wine')
print(len(drinks))
5
```

# 第六节。范围

内置的`range()`函数生成一个数字列表。`The format is:`

```
for number in range(3):
    print(number)0
1
2for number in range(1, 3):
    print(number)1
2for number in range(1, 10, 2):
    print(number)1
3
5
7
9drinks = ['water', 'tea', 'orange juice', 'beer']
for number in range(0, len(drinks), 2):
    print(drinks[number])
water
orange juice
```

# 第七节。结论

在本文中，您已经学习了如何在 Python 编程中使用列表。