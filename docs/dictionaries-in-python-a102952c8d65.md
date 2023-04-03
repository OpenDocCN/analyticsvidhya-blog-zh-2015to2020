# Python 中的字典

> 原文：<https://medium.com/analytics-vidhya/dictionaries-in-python-a102952c8d65?source=collection_archive---------17----------------------->

## 为了更好地理解，本文将提供一些片段。

![](img/20e05ad6cfa7736fbf614817bce13c03.png)

照片由[皮斯特亨](https://unsplash.com/@pisitheng?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

# 构建词典

在其他语言中，字典有时被称为“联想记忆”或“联想数组”。与序列不同，序列由一系列数字索引，字典由*键*索引，键可以是任何不可变的类型；字符串和数字总是可以作为键。如果元组只包含字符串、数字或元组，则可以用作键。如果一个元组直接或间接地包含任何可变对象，它就不能用作键。您不能使用列表作为键，因为列表可以使用索引赋值、片赋值或者像`append()`和`extend()`这样的方法来修改。

最好将字典视为一组*键:值*对，要求键是惟一的(在一个字典中)。一对括号创建了一个空字典:`{}`。将逗号分隔的键:值对列表放在大括号内会将初始键:值对添加到字典中；这也是字典在输出时的书写方式。

对字典的主要操作是用某个键存储一个值，并根据给定的键提取值。也可以用`del`删除一个键:值对。如果使用已经在使用的键进行存储，则与该键关联的旧值会被遗忘。使用不存在的键提取值是错误的。

对字典执行`list(d)`会返回字典中使用的所有键的列表，按插入顺序排列(如果您想对其进行排序，只需使用`sorted(d)`即可)。要检查字典中是否有单个键，使用`[in](https://docs.python.org/3.7/reference/expressions.html#in)`关键字。

下面是一个使用字典的小例子:

```
In[1] : my_dict = {'city':'New York','age':'21','city1':'Berlin'}In[2] :my_dict['city']Out[2]:'New York'In[3]: my_dict['city1']Out[3]:'Berlin'
```

值得注意的是，字典在数据类型上非常灵活。例如:

```
In [1] : my_dict **=** {'key1':123,'key2':[12,23,33],'key3':   ['item0','item1','item2']}
my_dict['key3'][0]Out[1]:'item0'In[2] :my_dict['key3'][0][2:4]Out[2]:'em'
```

简单地说，Python 有一个内置的自减或自加(或乘法或除法)方法。对于上面的语句，我们也可以使用+=或-=来表示。例如:

```
In[1] : # Set the object equal to itself minus 123 
        my_dict['key1'] -=   123
        my_dict['key1']Out[1] : -123
```

另外，注意 Python 是高度区分大小写的。

# 用字典嵌套

希望您开始看到 Python 在嵌套对象和调用对象方法方面的灵活性有多么强大。让我们看看嵌套在字典中的字典:

```
In[1] : # Dictionary nested inside a dictionary nested inside a              dictionary

       d = {'Country':{'state':{'city':'Pune'}}}Out[1]: 'Pune'In[2]: #Keep calling the keys
       d['Country']['state']['city']Out[2]: 'Pune'
```

# 一些字典方法

我们可以在字典上找到一些方法。让我们快速介绍其中的几个:

```
In[1] :# Create a typical dictionary
       d = {'key1':1,'key2':2,'key3':3}In [2]: # Create a typical dictionary
       d = {'key1':1,'key2':2,'key3':3}Out[2]: dict_keys(['key1', 'key2', 'key3'])In[3] : # Method to grab all values
       d.values()Out[3]: dict_values([1, 2, 3])In [4]:# Method to return tuples of all items  (we'll learn about tuples soon)
       d.items()Out[4]: dict_items([('key1', 1), ('key2', 2), ('key3', 3)])
```

希望您现在已经对如何构造字典有了很好的基本理解。