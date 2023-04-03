# 线性搜索和二分搜索法算法—数据结构

> 原文：<https://medium.com/analytics-vidhya/linear-search-and-binary-search-algorithm-data-structure-837a4581c4d0?source=collection_archive---------9----------------------->

![](img/61f650285c9a603eceaab274eb5c779a.png)

让我们开始—

1.  线性搜索—

假设我们有一个数组或列表[2，4，7，8，2，3，9]，你要在这个列表中搜索一个值。所以你会怎么做？

在这个算法中，我们取一个值，并与列表中的所有元素逐一比较，如果值存在，我们返回 found not not found。
所以，我们可以通过 for 循环逐个比较值。

```
**if** SearchItem(a,4):
    print(**'found'**)
**else**:
    print(**'not found'**)
```

在上面的代码中，我们检查 SearchItem()函数是否返回 true，然后元素存在，元素不存在。在 SearchItem()函数中，我们传递的是我们正在搜索的列表和元素。
所以，现在我们要制作 SearchItem()函数。

```
a=[2,3,4,5,4,6]
**def** SearchItem(list,item):
    **for** i **in** list:
        **if** i==item:
            **return True
    return False**
```

在上面的代码中，我们接受函数 SearchItem()中的列表和元素，然后我们检查 for 循环是否存在。

2.二分搜索法—

假设你有一个很长的列表，你要搜索的元素在列表的最后或者中间，那么你会怎么做？你可以用线性算法搜索，但是你的代码质量不好，因为你在所有的列表元素上迭代，最后你得到了值，所以你的运行时间有点高。
所以，对于这个问题我们用二分搜索法。在二分搜索法，你的清单应该是有序的。
假设你有一个数组=[2，3，5，6，4，56]。首先你要用 sorted()内置方法对它进行排序。那么列表将变成 array=[2，3，4，5，6，56]。在这个列表中，我们将第一个元素(最低值)称为 lower_bound，最后一个元素(最高值)称为 upper_bound。所以，

> lower_bound(l)=2，upper_bound(u)=56
> 指数 l=0，指数 u=5
> 现在我们将由(m) = (l+u)//2 求 mid 指数

现在，我们将查看搜索值是否小于 **array[m]** ，然后我们将“u”值更改为“m-1”(为什么是 m-1，因为我们已经检查了值索引 m)。现在我们有了新的“u”值，然后我们将在新的“u”值的帮助下找到“m ”,并再次进行相同的过程。

如果搜索值大于**数组[m]** ，那么我们将把‘l’值改为‘m+1’(为什么是 m+1，因为我们已经检查了值索引 m)。现在我们有了新的“l”值，然后我们将在新的“l”值的帮助下找到“m ”,并再次进行相同的过程。

现在看看这个例子，假设我们正在搜索 6。

> array=[2，3，4，5，6，56]
> 所以，l=0，u=5，m=(0+5)//2=2
> m 值为 array[m]=4
> 现在，搜索值为 6，大于 array[m]=4，所以我们将 l 改为 m+1，所以，现在 l=2+1，新的 m=3+5//2=4，u 也相同。
> 搜索值是 6，现在等于数组[m]=6，所以最后我们在列表中找到我们的值。
> 当搜索值小于数组[m]时，必须将 u 改为 m，向前的过程相同

现在让我们做一些代码—

```
list=[2,3,4,5,6,56]

**def** searchItem(list,n):
    list = sorted(list)#  sorting list

    l=0                 #initializing l value 
    u=len(list)-1       #initializing u value

    **while** l<=u:          # checking if l<=u then loop should run
        a=(int(l)+int(u))//2    #getting a value
        **if** list[a]==n:          # checking if that mid value is
                                #equal to searching value then 
                                # return True    
            **return True
        elif** list[a]<n:     # if list[a] value is less than 
                            # searching value then assigning new 
                            # value to l l = a+1
            **else**:               # if list[a] value is greater than 
                            # searching value then assigning new 
                            # value to u
            u=a-1
    return False**if** searchItem(list,n):
    print(**'found'**)
**else**:
    print(**'not found'**)
```

希望你能理解。

谢谢你。