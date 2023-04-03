# Python 列表- II

> 原文：<https://medium.com/analytics-vidhya/python-list-ii-48eefe298f01?source=collection_archive---------6----------------------->

# 在 Python 中将一个字符串拆分成一个列表

我们可以使用。split(separator)函数使用指定的分隔符将一个字符串分成一系列字符串。分隔符告诉 Python 在哪里断开字符串。如果没有指定分隔符，那么 Python 将使用字符串中的空白作为分隔点。

请参见下面的示例，该示例将一个文本字符串分解为单个单词:

```
my_text = ‘I am learning Python!’
words = my_text.split()  #splits the text wherever there is a space.
print(words)
```

上述程序产生以下输出:

```
[‘I’, ‘am’, ‘learning’, ‘Python!’]
```

用分隔符'-'将字符串拆分成列表的另一个示例:

```
my_string = ‘abcd-efgh-ijkl’
my_list = my_string.split(‘-’)
                      #split the text wherever there’s a hyphen(‘-')
print(my_list)
```

输出:

```
[‘abcd’, ‘efgh’, ‘ijkl’]
```

# Python String()函数

strip()方法删除字符串中的任何前导(开头)和尾随(结尾)字符(默认情况下，空格是字符)。

举个例子，

```
text = “    programming     “
my_text = text.strip() 
  #Removes the spaces at the beginning and at the end of the string.
print(my_text)
```

输出显示开头和结尾没有多余空格的字符串:

```
programming
```

# Python map()函数

Python map()函数用于在给定函数应用于 iterable(列表、元组等)的每个项目后返回结果列表。

让我们看一个例子:

**使用 map()函数获取用户输入的数字列表:**

```
x = list(map(int,input(“Enter the elements“).strip().split(‘ ‘)))
                # Reads number inputs from user using map() function
print("List is — “, x)
```

输出:

```
Enter the elements: 10 20 30 40
List is — [10, 20, 30, 40]
```

您会发现这个函数在竞争激烈的编码环境中非常有用。

# 遍历列表

我们可以使用 for 循环遍历列表中的元素。举个例子，

```
my_list = [‘pizza’, ‘sandwich’, ‘salad’]
for x in my_list:
    print(“I like ”, x)
```

输出将是:

```
I like pizza
I like sandwich
I like salad
```

# 检查列表中是否存在元素

我们可以使用关键字“in”来测试一个元素是否存在于一个列表中。

参见下面的例子，

```
my_list = [‘pizza’, ‘sandwich’, ‘salad’]
if ‘pizza’ in my_list:     #checks if 'pizza' is present in the list
    print(“Yes, pizza is present in my list.”)
else:
    print(“No, pizza is not present in my list.”)
```

输出:

```
Yes, pizza is present in my list.
```

# 列表的长度

我们可以使用 len()函数来确定列表的长度，即列表中元素的数量。

```
my_list = [‘hamburger’, ‘pizza’, ‘bread’, ‘fish’]
print(“Number of elements in my list: ”, len(my_list))
```

并且，输出给出:

```
Number of elements in my list: 4
```

# 从列表中删除元素

有几种方法可以从列表中删除项目:

(i) remove()方法—从列表中删除指定的元素

```
my_list =[‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
my_list.remove(‘chocolate’)
print(my_list)
```

指定的元素“chocolate”从列表中删除，输出变为:

```
[‘pizza’, ‘hamburger’, ‘milk’]
```

(ii) pop()方法—删除给定索引处的元素

如果没有提供索引，pop()方法将删除列表中的最后一个元素。举个例子，

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
print(my_list.pop())
```

最后一个元素“milk”从列表中删除，输出变为:

```
[‘pizza’, ‘chocolate’, ‘hamburger’]
```

(iii) del 关键字—删除指定索引处的元素

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
del my_list[1]
print(my_list)
```

移除索引 1 处的元素。

输出:

```
[‘pizza’, ‘hamburger’, ‘milk’]
```

del 关键字也可以完全删除列表:

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
del my_list
print(my_list)    
        #this will cause an error because we have deleted “my_list”.
```

(iv) clear()方法—清空列表

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
my_list.clear()
print(my_list)
```

所有元素都被删除，列表变成一个空列表。因此输出将是:

```
[]
```

# 复制列表

有多种方法可以制作列表的副本，一种方法是使用内置的 list 方法 copy()。

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
copy_list = my_list.copy()
print(copy_list)
```

制作副本的另一种方法是使用内置的方法列表()。

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
copy_list = list(my_list)
print(copy_list)
```

创建并打印列表 my_list 的副本。

上述程序的输出将是:

```
[‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
```

# 对列表进行排序

默认情况下，sort()方法按升序对列表进行排序。

检查下面的例子，

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
my_list.sort()
print(my_list)
```

输出:

```
[‘chocolate’, ‘hamburger’, ‘milk’, ‘pizza’]
```

要按降序对列表进行排序:

```
my_list = [‘pizza’, ‘chocolate’, ‘hamburger’, ‘milk’]
my_list.sort(reverse=True)
print(my_list)
```

输出将是:

```
[‘pizza’, ‘milk’, ‘hamburger’, ‘chocolate’]
```