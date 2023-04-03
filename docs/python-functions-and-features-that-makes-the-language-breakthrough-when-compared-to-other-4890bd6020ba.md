# 与其他语言相比，Python 函数和特性是语言的突破

> 原文：<https://medium.com/analytics-vidhya/python-functions-and-features-that-makes-the-language-breakthrough-when-compared-to-other-4890bd6020ba?source=collection_archive---------34----------------------->

![](img/5ec334fa2b29d22a09464ddcd62bd2bc.png)

来自[突发事件](https://burst.shopify.com/api-work-productivity?utm_campaign=photo_credit&amp;utm_content=Free+Software+Programming+Plan+Photo+%E2%80%94+High+Res+Pictures&amp;utm_medium=referral&amp;utm_source=credit)的[莎拉·普鲁格](https://burst.shopify.com/@sarahpflugphoto?utm_campaign=photo_credit&amp;utm_content=Free+Software+Programming+Plan+Photo+%E2%80%94+High+Res+Pictures&amp;utm_medium=referral&amp;utm_source=credit)的照片

在现代，计算机科学及其众多的子学科与现实世界中的许多案例相结合，编程语言在计算机科学的发展中起着非常重要的作用，我认为编程语言是任何计算机科学问题的骨架。因此，拥有一种全功能的编程语言为计算机科学开发者解决问题提供了更多的控制。Python 就是这样一种编程语言，它提供了各种方便用户的工具和功能。在 stack overflow 2020 开发者调查中，它也被评为最受欢迎和第三受欢迎的语言。因此，从调查结果来看，它可以理解有大量的社区支持 python，这有助于它更快地发展。

在考虑了 python 的使用和上下文之后，让我们来讨论这篇文章。在这里，我们将介绍帮助 python 获得这些调查结果的各种方法和函数。此外，这些功能和工具也将帮助读者进行竞争性编码。

# **1。计数器:-**

计数器类是 Python 集合模块提供的一种特殊类型的数据集对象。集合模块为用户提供了专门的容器数据类型，从而为 Python 的通用内置(如字典、列表和元组)提供了一种替代方案。它返回一个字典，这个字典查找数据结构中特定元素的出现。就像下面的例子，当列表被传递给计数器时，它返回一个字典，该字典将关键字作为列表项，关键字的值是它们在列表中出现的值。

```
#importing counter
from collections import Counterl=[1,2,3,3,435,5,67,666,11,11,1,1]
c=Counter(l)
print(c)
```

输出:

```
[out]:Counter({1: 3, 3: 2, 11: 2, 2: 1, 435: 1, 5: 1, 67: 1, 666: 1})
```

计数器的 most_common 函数示例，
为了方便用户，计数器附带了许多函数，但对我来说，most_common 函数非常有用，它从提供的数据结构中返回出现次数最高的项。

在下面给出的示例中，most_common 方法应用于上面示例中生成的计数器。传递 2 是因为参数意味着我们只需要出现次数最多的两个项目。它返回具有 2 个元素元组的列表，元组的第一个元素指示关键字，元组的第二个元素指示该关键字的出现。如下例所示，在返回列表的第 0 个索引上，出现了两个元素的元组，其中指示出现了 3 次的键 1。

```
*#this provides most common two elements that are 1 which is 3 times and 3 which is two times* c.most_common(2)
```

输出:

```
[out]: [(1,3),(3,2)]
```

2 .枚举:-

枚举函数在 for 循环中很有帮助，因为在 for 循环中需要 iterable 元素的索引。在某些用例中进行竞争性编码时，需要用值维护 iterable 的索引，因此为了执行上述任务，python 具有枚举功能。

在下面的代码片段中,“I”关键字维护索引，而“j”维护索引 I 处的值。

```
l=[1,42,990,5,7,8,64]
**for** i,j **in** enumerate(l):    
    print(str(i)+"------->"+str(j))
```

输出:

```
[out]:0------->1
1------->42
2------->990
3------->5
4------->7
5------->8
6------->64
```

**3.zip :-**

当需要组合两个或多个数据结构的各个元素时，使用 zip 函数，它将两个或多个 iterable 作为参数

在下面的例子中，我们有两个列表 l 和 k，当我们将这两个列表传递给 zip 函数时，它返回一个迭代器，当显示为列表时，其中包含几个两元素元组，元组的第一个元素来自第一个列表，元组的第二个元素来自第二个列表。

```
l=[1,2,3]
k=[4,5,6]
a=zip(l,k)
print(a)*#returns iterator object*
print(list(a))*#clubbed both list*
```

输出:

```
[out]:<zip object at 0x0000020198B6F908>
[(1, 4), (2, 5), (3, 6)]
```

**4 .地图:-**

当您需要对 iterable 的每个元素应用任何特定的函数时，可以使用 map 函数

如果我们想计算一个列表中所有元素的平方，那么我们可以在 python 中只用一行代码来完成。如下例所示，在 map 函数中传递的两个参数中，第一个是我们希望应用于列表中每个元素的函数，第二个是列表本身。这里我们使用 lambda 函数作为参数，我们将在本文后面讨论。

```
l=[1,2,3] 
a=map(**lambda** x:x**2,l) 
print(a)*#returns iterator object* 
print(list(a))*#updated list*
```

输出:

```
[out]:<map object at 0x0000020198C93BC8>
[1, 4, 9]
```

5 .过滤器:-

这与 map 函数的工作方式类似，但它包括条件功能。因此，如果我们想只对列表中的选定元素应用特定的函数，而不是像在 map 中那样对所有元素应用，那么就要使用过滤器。Filter 函数还接受两个参数，其中第一个参数是函数，如果为列表中的元素返回“True ”,则只考虑该特定元素，否则它将被丢弃。

```
**def** is_even(i):     
   **return** i%2==0      
l=[1,2,3,4] 
filtered=filter(is_even,l) 
print(filtered)*#filtered iterator object* print(list(filtered))*#filtered list*
```

输出:

```
[out]:<filter object at 0x0000020199150448>
[2, 4]
```

**6。__ 词典 __**

这对于 python 中面向对象的开发人员来说是一个福音。它以字典形式
给出了所有实例属性的信息，如下例所示。如果我们想要任何对象的所有详细信息，我们必须调用 __dict__ 方法来显示它的所有变量和值。

```
class emplyoee:
    def __init__(self,name,age,pay):
        self.name=name
        self.age=age
        self.pay=pay
emp=emplyoee("devarsh",20,100000)
print(emp.__dict__)*#printing all attribute*
```

输出:

```
[out]: {'name': 'devarsh', 'age': 20, 'pay': 100000}
```

**7。已命名的对:-**

如果您需要访问字典形式的元组，您可以提供一个键来在元组中进行搜索，那么命名元组是最好的方法。因此，命名元组包含了元组和字典的优点。在命名元组中，我们可以用一个键搜索元组中的一个元素。
在下面的例子中，首先在导入命名元组之后，我们声明了命名元组“dev ”,它以列表格式存储年龄和工资。之后，我们通过创建一个实例“d”来添加名为 tuple 的“dev”中的值。现在，在“d”的帮助下，我们可以搜索我们在实例化“age”键时输入的年龄。

```
from collections import namedtuple
#declaring named tuplesdev = namedtuple("Dev",["age","pay"])
d = dev(20,100000)
print(d.age)  *#we can access tuple with the key*
print(d[1])
```

输出:

```
[out]: 20 100000
```

排序:-

Sorted 是 python 中主要用于非就地排序的函数。通过保持 reverse 等于 TRUE，我们可以对提供的列表进行逆序排序。排序函数的主要用途是借助 key 参数，因为使用它我们可以对 iterable 的内部元素进行排序。例如，我们可以在包含列表中各种元组的数据结构中根据元组索引进行排序。

在排序函数中，键参数被定义为控制所提供列表排序的函数。因此，我们可以在关键参数的帮助下执行自定义排序

```
l=[1,2,5,4,7,8,96,3,2,1,4,50,56] 
print("Sorted")
print(sorted(l))*#returns another sorted list*print("original list") 
print(l)*#first list not sorted as it is* l=[(2,3,4),(4,7,5),(5,8,3),(8,4,1)] *#if we want above list to be sorted with respect to second element of tuple then* 
print("Sorting according to second element of the tuple in reverse")print(sorted(l,reverse=**True**,key=**lambda** x:x[1]))*#prints sorted list according to second element of tuple in descending order*
```

输出:

```
[out]: Sorted
[1, 1, 2, 2, 3, 4, 4, 5, 7, 8, 50, 56, 96]
original list
[1, 2, 5, 4, 7, 8, 96, 3, 2, 1, 4, 50, 56]
Sorting according to second element of the tuple in reverse
[(5, 8, 3), (4, 7, 5), (8, 4, 1), (2, 3, 4)]
```

**9 .使用键**的最小最大值

列表中的 Min 和 max 函数只是用来从 iterable 中获取最小值和最大值，但是在 key 函数的帮助下，我们可以根据情况需要提取 min 和 max。
假设我们需要根据位于列表不同索引处的字符串的长度提取最小值和最大值，下面的代码用于实现这一点。

```
l=["devarsh","h","patel"] 
print("print the maximum length string ") 
print("---->"+str(max(l,key=**lambda** x:len(x)))) 
print("print the minimum length string ") 
print("---->"+str(min(l,key=**lambda** x:len(x))))
```

输出:

```
[out]:print the maximum length string  
---->devarsh 
print the minimum length string  
---->h
```

**10。默认词典:-**

我们需要从集合模块中导入默认字典，这是一个与字典类似的数据结构，但是如果我们请求字典中不存在的键，那么它不会返回键错误，但不会返回任何错误。当有很高的可能性从用户端搜索关键字时，使用 Defaultdict，而不是在字典中。因此，这可以防止键错误并阻止 python 程序暂停。Defaultdict 在参数中起作用，返回键不存在时要显示的消息。

```
**from** **collections** **import** defaultdict
**def** msg():
    **return** "not present"
d=defaultdict(msg)
d["name"]="Devarsh"
d["surname"]="patel"
print(d["name"])
print(d["age"])*#age is not present so it will return the not present from the msg function*
```

输出:

```
[out]:Devarsh
not present
```

**11。Re 库(正则表达式):-**

正则表达式对于文本处理非常重要，因此 python 附带了用于此目的的 re 库
其背后的基本思想是创建模式，然后启动查询在字符串中搜索该模式。因此，re 库中的基本概念是通过编译各种正则表达式来创建模式，就像下面的表达式一样，我们编译模式来从句子中查找单词。之后，通过将模式作为参数与 string 一起传递，可以将 find 和 finditer 等各种方法应用于字符串。这里我应用了 finditer 方法来查找单词，因为它返回带有索引跨度的单词。

```
st="My name is devarsh h patel.Hello all how are  you this is number to test 12456789 and special character too !@ # $%"
print("All words in the string")
pattern=re.compile(r"\w+")*#/w+ is a regular expression to find words*
match=re.finditer(pattern,st)*#find iter returns span that is index too so it is useful*
**for** i **in** match:
    print(i)
```

输出:

```
[out]: All words in the string
<re.Match object; span=(0, 2), match='My'>
<re.Match object; span=(3, 7), match='name'>
<re.Match object; span=(8, 10), match='is'>
<re.Match object; span=(11, 18), match='devarsh'>
<re.Match object; span=(19, 20), match='h'>
<re.Match object; span=(21, 26), match='patel'>
<re.Match object; span=(27, 32), match='Hello'>
<re.Match object; span=(33, 36), match='all'>
<re.Match object; span=(37, 40), match='how'>
<re.Match object; span=(41, 44), match='are'>
<re.Match object; span=(46, 49), match='you'>
<re.Match object; span=(50, 54), match='this'>
<re.Match object; span=(55, 57), match='is'>
<re.Match object; span=(58, 64), match='number'>
<re.Match object; span=(65, 67), match='to'>
<re.Match object; span=(68, 72), match='test'>
<re.Match object; span=(73, 81), match='12456789'>
<re.Match object; span=(82, 85), match='and'>
<re.Match object; span=(86, 93), match='special'>
<re.Match object; span=(94, 103), match='character'>
<re.Match object; span=(104, 107), match='too'>
```

**12。λ函数:-**

这个函数也称为匿名函数，这个函数没有名字，它接受参数，并在一行代码中返回结果。具体来说，它用于定义排序和相似函数的键
冒号的左边是函数的参数，右边是函数的返回值。

```
*#function to calculate the number is even or not*
*#normal function*
print("Normal function")
**def** is_even(i):
    **return** i%2==0
print(is_even(1))
*#with the help of lambda*
print("lambda function")
x=**lambda** s:s%2==0
print(x(2))
```

输出:

```
[out]:Normal function
False
lambda function
True
```

**13。装饰师**

Decorators 为开发人员提供了更改或添加函数的能力，一些代码行甚至不需要更改函数的源代码。
decorator 主要用于开发人员想要更新库中某个功能的时候。在下面的例子中，我们增加了计算函数执行时间的功能，甚至不需要修改函数的任何代码。

```
*# time calculating decorator*
**import** **time**
**def** t(fun):
    **def** wrapper(q):
        t1=time.time()
        returned=fun(q)
        **for** i **in** range(1,1000000): *# to increase run time*
            **pass**    
        t2=time.time()
        time_taken=t2-t1
        print("time taken to execute function  "+str(time_taken))
        **return** returned
    **return** wrapper

@t
**def** square(q):
    **return** q**2
print("The square of the number is " + str(square(4)))
```

输出:

```
[out]:time taken to execute function  0.021939992904663086
The square of the number is 16
```

**14。发电机**

生成器是迭代器，帮助使用实例化数据，而不是使用整个数据。此任务防止系统加载全部数据，而是加载唯一需要的部分。
主要是当用户需要在程序中处理数百万数据而不耗尽内存时使用生成器。

创建发生器有两种方法:

1.借助 yield 关键字

```
*#generator function*
**def** g():
    **for** i **in** range(1,11):
        **yield** i
print(g)*#ierator of genartor created*
**for** i **in** g():
    print(i)
```

输出:

```
[out]:<function g at 0x0000020198D6B5E8> 
1 2 3 4 5 6 7 8 9 10
```

2.生成器理解

```
*#generator compprehension*
g=(i **for** i **in** range(1,11))*#generator created*
print(g)*#ierator of genartor created*
**for** i **in** g:
    print(i)
```

输出:

```
[out]:<generator object <genexpr> at 0x0000020199174D48> 
1 2 3 4 5 6 7 8 9 10
```

**15。*参数:-**

当我们需要向函数传递任意数量的参数时，我们将函数参数定义为*args，它接受任意数量的参数并将其转换为元组。当传递给函数的参数数量没有预先定义时，使用 args。就像下面的例子一样，在计算平方的函数中，我们分别传递一个两个和三个参数，函数对所有三个值都能准确地工作。

```
**def** square(*args):
    print(type(args))
    **return** [i**2 **for** i **in** args]
*#passing different number of argument*
print(square(1,2,3))
print(square(5,7))
print(square(9))
```

输出:

```
[out]:<class 'tuple'> 
[1, 4, 9] 
<class 'tuple'> 
[25, 49] 
<class 'tuple'> 
[81]
```

**16**kwargs**

当我们需要将字典类型的数据结构参数传递给函数时，我们将函数参数定义为**kwargs，它将字典类型的数据结构作为参数，并将其转换为纯字典

```
**def** info(**kwargs):
    **for** i,j **in** kwargs.items():
        print(str(i)+"--->"+str(j))
print(info(name="devarsh",surname="patel"))
```

输出:

```
[out]:name--->devarsh 
surname--->patel
```

**17。使用集合从列表或任何合适的可重复项中删除重复值**

```
l=[1,1,1,1,1,2,2,2,3,3,3,5,4,7,5] 
l=set(l)*#removed dupllicate* 
print(list(l))
```

输出:

```
[out]:[1, 2, 3, 4, 5, 7]
```

**18。所有功能**

如果列表中的所有值都为真，则 All 函数返回真

```
l=[i%2==0 **for** i **in** range(1,11)] *#appends true if even else false* 
print(l)
print(all(l))
```

输出:

```
[out]:[False, True, False, True, False, True, False, True, False, True] False
```

**19。任何功能**

如果列表中的任何值为真，则任何函数都返回真

```
l=[i%2==0 **for** i **in** range(1,11)]  *#appens true if even else false* 
print(l)
print(any(l))
```

输出:

```
[out]:[False, True, False, True, False, True, False, True, False, True]
True
```

# **结论**

这个博客的基本目的是让读者了解 python 晦涩的功能。通过了解这些函数，用户可以使用 python 以有效的方式产生更多的结果。因此，在这篇博客中，我们讨论了 python 中的各种工具和函数，这将有助于初学者更好地以实用的方式理解 python 语言工具。

对于博客的完整代码访问:[https://github.com/Devarsh23/Python-Tools-and-functions](https://github.com/Devarsh23/Python-Tools-and-functions)