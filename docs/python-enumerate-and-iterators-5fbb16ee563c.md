# Python 枚举器和迭代器

> 原文：<https://medium.com/analytics-vidhya/python-enumerate-and-iterators-5fbb16ee563c?source=collection_archive---------27----------------------->

Python 附带了许多易于使用的内置函数。这篇文章旨在讨论`enumerate`函数，并简要介绍如何使用迭代器。

和往常一样，让我先给你源代码的链接:

[](https://github.com/python/cpython/blob/master/Python/bltinmodule.c) [## python/cpython

### Python 编程语言。通过在 GitHub 上创建帐户，为 python/cpython 开发做出贡献。

github.com](https://github.com/python/cpython/blob/master/Python/bltinmodule.c) [](https://github.com/python/cpython/blob/master/Objects/enumobject.c) [## python/cpython

### Python 编程语言。通过在 GitHub 上创建帐户，为 python/cpython 开发做出贡献。

github.com](https://github.com/python/cpython/blob/master/Objects/enumobject.c) 

这是来自[文档](https://docs.python.org/3/library/functions.html#enumerate)的函数定义:

`enumerate(*iterable*, *start=0*)`

其中`iterable`是一个序列，一个迭代器或者一个支持迭代的对象。
`start`为整数，默认为`0`。

以下是您可以想到的实现枚举的方式:

```
def enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1
```

以下是一些如何使用枚举的示例:

```
months = [**"January"**, **"February"**, **"March"**, **"April"**, **"May"**, **"June"**, **"July"**, **"August"**, **"September"**, **"October"**, **"November"**, **"December"**]

**for** month_no, month **in** enumerate(months, start=1):
    print(month_no, **" - "**, month)

print(**"Months in Reverse Order:"**)
**for** month_no, month **in** enumerate(months[::-1], start=-12):
    print(month_no, **" - "**, month)Output:
*******
1  -  January
2  -  February
3  -  March
4  -  April
5  -  May
6  -  June
7  -  July
8  -  August
9  -  September
10  -  October
11  -  November
12  -  DecemberMonths in Reverse Order:
-12  -  December
-11  -  November
-10  -  October
-9  -  September
-8  -  August
-7  -  July
-6  -  June
-5  -  May
-4  -  April
-3  -  March
-2  -  February
-1  -  January
```

嗯，`start`只支持整数，所以如果你想要文本标签或其他东西，你可以这样做:

```
**import** string
things_to_do = [**"get up"**, **"exercise"**, **"eat"**, **"pray"**, **"study"**, **"have fun"**, **"sleep"**]
**for** label, activity **in** enumerate(things_to_do, start=0):
    print(string.ascii_lowercase[label], **") "**, activity) Output:
*******a )  get up
b )  exercise
c )  eat
d )  pray
e )  study
f )  have fun
g )  sleep
```

如你所见,`start`可以是负整数也可以是正整数，但它只能是一个`integer`

下面的例子展示了如何使用迭代器而不是序列:

```
**class** CalendarMonths:

    **def** __init__(self, wrap_around=**False**, start_month=**None**):
        self.months = [**"January"**, **"February"**, **"March"**, **"April"**, **"May"**, **"June"**, **"July"**, **"August"**, **"September"**, **"October"**, **"November"**, **"December"**]
        self.wrap_around = wrap_around
        self.start_month = start_month

    **def** __iter__(self):
        **if** self.start_month:
            **try**:
                self.cur_index = self.months.index(self.start_month.capitalize()) - 1
            **except** ValueError **as** ve:
                print(**"%s is not a valid month"**)
                self.cur_index = -1
            **except** Exception **as** ex:
                print(**"Exception when trying to process start month %s. Error : %s"** % (self.start_month, ex))
                self.cur_index = -1
        **else**:
            self.cur_index = -1
        **return** self

    **def** __next__(self):
        self.cur_index += 1
        cur_index_exceeds_item_length = self.cur_index > len(self.months) - 1
        self.cur_index = 0 **if** self.wrap_around **and** cur_index_exceeds_item_length **else** self.cur_index
        **if not** self.wrap_around **and** cur_index_exceeds_item_length:
            **raise** StopIteration(**"No more months to display"**)
        **return** self.months[self.cur_index]

**def** print_months(months_iterator, r, start):
    **try**:
        **for** i, _ **in** enumerate(range(r), start=start):
            print(**"Call No - "**, i, **" - "**, next(months_iterator))
    **except** Exception **as** ex:
        print(ex)

months_with_wrap_around = CalendarMonths(**True**)
months_without_wrap_around = CalendarMonths(**False**)
months_with_wrap_around_start_at_march = CalendarMonths(**True**, **"march"**)
months_without_wrap_around_start_at_march = CalendarMonths(**False**, **"march"**)

**for** obj **in** [months_with_wrap_around, months_without_wrap_around, months_without_wrap_around_start_at_march, months_with_wrap_around_start_at_march]:
    print_months(iter(obj), 25, 1)
```

输出将是这样的:

```
Call No -  1  -  January
Call No -  2  -  February
Call No -  3  -  March
Call No -  4  -  April
Call No -  5  -  May
Call No -  6  -  June
Call No -  7  -  July
Call No -  8  -  August
Call No -  9  -  September
Call No -  10  -  October
Call No -  11  -  November
Call No -  12  -  December
Call No -  13  -  January
Call No -  14  -  February
Call No -  15  -  March
Call No -  16  -  April
Call No -  17  -  May
Call No -  18  -  June
Call No -  19  -  July
Call No -  20  -  August
Call No -  21  -  September
Call No -  22  -  October
Call No -  23  -  November
Call No -  24  -  December
Call No -  25  -  January
Call No -  1  -  January
Call No -  2  -  February
Call No -  3  -  March
Call No -  4  -  April
Call No -  5  -  May
Call No -  6  -  June
Call No -  7  -  July
Call No -  8  -  August
Call No -  9  -  September
Call No -  10  -  October
Call No -  11  -  November
Call No -  12  -  December
No more months to displayCall No -  1  -  March
Call No -  2  -  April
Call No -  3  -  May
Call No -  4  -  June
Call No -  5  -  July
Call No -  6  -  August
Call No -  7  -  September
Call No -  8  -  October
Call No -  9  -  November
Call No -  10  -  December
No more months to displayCall No -  1  -  March
Call No -  2  -  April
Call No -  3  -  May
Call No -  4  -  June
Call No -  5  -  July
Call No -  6  -  August
Call No -  7  -  September
Call No -  8  -  October
Call No -  9  -  November
Call No -  10  -  December
Call No -  11  -  January
Call No -  12  -  February
Call No -  13  -  March
Call No -  14  -  April
Call No -  15  -  May
Call No -  16  -  June
Call No -  17  -  July
Call No -  18  -  August
Call No -  19  -  September
Call No -  20  -  October
Call No -  21  -  November
Call No -  22  -  December
Call No -  23  -  January
Call No -  24  -  February
Call No -  25  -  MarchProcess finished with exit code 0
```

所以你知道了`enumerate`和`Iterators`时间，去和他们玩得开心。享受学习。

***花絮:*** 计算机科学中最奇怪的事情之一是为什么所有的索引都从 0 开始，而不是从 1 开始(谷歌一下其中的原因，你就会明白为什么从性能的角度来看这是必要的)