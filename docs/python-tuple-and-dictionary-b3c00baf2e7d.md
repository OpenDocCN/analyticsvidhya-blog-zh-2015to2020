# Python 元组和字典

> 原文：<https://medium.com/analytics-vidhya/python-tuple-and-dictionary-b3c00baf2e7d?source=collection_archive---------11----------------------->

在这一节中，我们来学习 Python 元组和字典。

# Python 元组

元组是有序且不可改变的集合。

元组在所有方面都与列表相同，除了以下属性:

*   元组是通过将元素括在括号( )而不是方括号[ ]中来定义的。
*   元组是不可变的(元组的元素一旦被赋值就不能被改变)。

# 创建元组

元组是通过将所有元素放在括号( )内，用逗号分隔来创建的。括号是可选的，但是使用它们是一个好习惯。一个元组可以有任意数量的项，它们可以是不同的类型(整数、浮点、列表、字符串等)。).

示例:

```
my_tuple = ();   //Empty tuplemy_tuple = (‘hello’,)
//For creating tuples with a single element, add a trailing comma     to indicate that it is a tuple.my_tuple = (10, ‘cat’, 99.9, (11, 22, 33)) 
               //Nested tuple with mixed data type
```

您已经学习并在列表上执行的各种操作，例如访问元素、索引和负索引、切片、遍历元素、元组长度、检查元素是否存在以及排序，对于元组也是如此。

但是元组是不能修改的。诸如追加、扩展、改变元素和从元组中移除元素的操作是不可能的，并且最终会导致错误。

# 删除元组

使用 del 关键字可以完全删除一个元组。举个例子，

```
my_tuple = (‘cat’, ‘rat’, ‘fish’)
del my_tuple           //my_tuple gets deleted completely.
```

# 元组方法

Python 有两个内置方法 count()和 index()，您可以对元组使用这两个方法。

## count()方法

count()方法返回指定值在元组中出现的次数。

```
my_tuple = (30, 55, 23, 99, 21, 55, 23, 55, 42, 61, 99, 21, 55)
x = my_tuple.count(55)
    //Returns the number of times the value 55 appears in the tuple.
print(x)
```

输出给出:

```
4
```

## index()方法

index()方法查找指定值的第一个匹配项。如果找不到该值，它将引发异常。这是一个例子，

```
my_tuple = (30, 55, 23, 99, 21, 55, 23, 55, 42, 61, 99, 21, 55)
x = my_tuple.index(23) 
     //Searches for the first occurrence of the value 23, and returns its position.
print(x)
```

输出将是:

```
2
```

因此，当您不想修改数据时，可以使用元组。如果集合中的值在程序的整个生命周期中保持不变，使用元组可以防止意外修改。

# Python 词典

Python 字典是一个无序的、可变的、索引的集合。字典中的每个条目都有一个键:值对，写在用逗号分隔的大括号中。值可以重复，但键必须是唯一的，并且必须是不可变的类型(字符串、数字或元组)。

让我们看一些例子:

```
my_dict = {} //empty dictionarymy_dict = {1: ‘pizza’, 2: ‘burger’, 3: ‘milk’} 
            //dictionary with integer keysmy_dict = {‘name’: ‘Katie’, 1: [10, 20, 30], ‘dateofbirth’:1994 }
            //dictionary with mixed keys
```

# 从字典中访问元素

您可以通过引用方括号中的关键字名称来访问字典的元素。

```
my_dict = {‘name’: ‘Alexa’, ‘age’: 20, ‘place’: ‘Canada’}
print(my_dict[‘age’])
```

还有一个名为 get()的方法来访问元素。

```
my_dict = {‘name’: ‘Alexa’, ‘age’: 20, ‘place’: ‘Canada’}
print(my_dict.get(‘age’))
```

以上两个程序都给出了以下结果:

```
20
```

# 更改元素的值

您可以通过引用特定元素的键名来更改其值。

```
my_dict = {‘name’: ‘Alexa’, ‘age’: 20, ‘place’: ‘Canada’}
my_dict[‘place’] = ‘Australia’
```

字典‘my _ dict’变成了:

```
{‘name’: ‘Alexa’, ‘age’: 20, ‘place’: ‘Australia’}
```

# 在字典中循环

您可以使用 for 循环遍历字典。

当遍历字典时，返回值是字典的键。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
for x in my_dict:
    print(x)         //print all keys in the dictionary one by one
```

输出:

```
name
place
age
```

**注意:**字典是无序的集合。也就是说，字典的元素并不是以任何特定的顺序存储的。因此，元素不会按照创建时的顺序进行迭代。

要逐一打印字典中的所有值:

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
for x in my_dict:
    print(my_list[x])
```

输出:

```
Mariam
London
20
```

可以使用 items()方法循环访问键和值。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
for x, y in my_dict.items():
    print(x, “:”, y)
```

输出:

```
place : London
name : Mariam
age : 20
```

# 检查密钥是否存在

若要检查字典中是否存在指定的键，请使用“in”关键字。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
if ‘name’ in my_dict:
    print(“Yes, name is one of the keys in my dictionary.”)
```

这将提供:

```
Yes, name is one of the keys in my dictionary.
```

# 字典长度

要确定一个字典有多少个元素(键值对)，可以使用 len()函数。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
print(len(my_dict))
```

这给出了输出:

```
3
```

# 向字典中添加元素

通过使用新的索引键并为其赋值，可以将元素添加到字典中。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
my_dict[‘phone’] = 688143
print(my_dict)
```

输出变成:

```
{‘age’: 20, ‘name’: ‘Mariam’, ‘phone’: 688143, ‘place’: ‘London’}
```

# 从字典中删除元素

从字典中删除元素有多种方法:

(I)pop()-删除具有指定键名的元素。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
my_dict.pop(‘age’)      //Removes the element with the key ‘age’
print(my_dict)
```

输出:

```
{‘name’: ‘Mariam’, ‘place’: ‘London’}
```

(ii)删除-删除具有指定键名的元素。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
del my_dict[‘age’]
print(my_dict)
```

输出:

```
{‘name’: ‘Mariam’, ‘place’: ‘London’}
```

del 关键字也可以完全删除字典。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
del my_dict           //my_dict gets deleted completely.
```

(iv) clear() —清空字典。

```
my_dict = {‘name’: ‘Mariam’, ‘age’: 20, ‘place’: ‘London’}
my_dict.clear()
print(my_dict)
```

这将产生一个空字典:

```
{}
```