# 排序算法在 Python 中的实现

> 原文：<https://medium.com/analytics-vidhya/implementation-of-sorting-algorithms-in-python-3eb100caedcb?source=collection_archive---------25----------------------->

![](img/07aed1624a815eca492236e74c0ed90f.png)

凯利·西克玛在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

计算机科学中三种最流行但被低估的算法是合并排序、快速排序和堆排序。听起来矛盾？怎么会有受欢迎却被低估的东西呢？是的，这就是我要解释的。几乎所有计算机科学的学生都用 Sudo 代码研究和学习了这三种算法，但很少有学生实现它们。这可能是因为我们关心的是算法，以及这些排序算法的运行时间，而不是实现，这仍然很好。

有大量的材料和参考资料来深入描述/Sudo 代码和运行时的细节。在这里，我将更多地关注实现部分，而不是算法。并且，我将使用 python 来实现。

这三种算法非常强大，并且在计算算法方面具有多种特性。他们提出了分而治之的策略以及贪婪的方法。

让我们用 Python 实现一个堆排序算法。注*堆可以是最小堆或最大堆。在这里，我们将集中在最小堆，但除了你需要改变操作符'【T4]'或反之亦然，这两种情况都适用。

```
**import** math

*# You cannot add any items to the heap after you start deleting the node***def** check_min_index(nodelst, child_index_1, child_index_2):
    **if** nodelst[child_index_1] >= nodelst[child_index_2]:
        **return** child_index_2
    **if** nodelst[child_index_1] < nodelst[child_index_2]:
        **return** child_index_1*# Recursive way* **def** check_bubble_up(nodelst, index):
    **if** index == 0:
        **return True** parent = math.floor(index/2)
    **if** nodelst[parent] > nodelst[index]:
        tmp = nodelst[index]
        nodelst[index] = nodelst[parent]
        nodelst[parent] = tmp
    check_bubble_up(nodelst, parent)
    **return True** *# iterative way
# def check_bubble_up(nodelst, index):
#     while index != 0:
#         parent = math.floor(index/2)
#         if nodelst[parent] > nodelst[index]:
#             tmp = nodelst[index]
#             nodelst[index] = nodelst[parent]
#             nodelst[parent] = tmp
#         index = parent
#     return True***def** check_bubble_down(nodelst, cur_idx, last_idx):
    **if** last_idx == cur_idx:
        **return True** child_index_1 = 2 * cur_idx + 1
    child_index_2 = 2 * cur_idx + 2
    **if** (child_index_1 >= last_idx) **and** (child_index_1 > last_idx):
        **return True
    if** (child_index_1 <= last_idx) **and** (child_index_1 >= last_idx):
        min_idx = child_index_1
        **if** nodelst[cur_idx] > nodelst[min_idx]:
            tmp = nodelst[cur_idx]
            nodelst[cur_idx] = nodelst[min_idx]
            nodelst[min_idx] = tmp
        check_bubble_down(nodelst, min_idx, last_idx)
    **else**:
        min_idx = check_min_index(nodelst, child_index_1, child_index_2)
        **if** nodelst[cur_idx] > nodelst[min_idx]:
            tmp = nodelst[cur_idx]
            nodelst[cur_idx] = nodelst[min_idx]
            nodelst[min_idx] = tmp
        check_bubble_down(nodelst, min_idx, last_idx)**class** Heap:
    **def** __init__(self):
        self.nodelst = []
        self.last_pos = -1

    **def** add_node(self, node):
        self.nodelst.append(node)
        self.last_pos += 1
        check_bubble_up(self.nodelst, len(self.nodelst)-1)

    **def** delete_node(self):
        heap_size = len(self.nodelst)
        **if** heap_size != 0:
            tmp = self.nodelst[0]
            self.nodelst[0] = self.nodelst[self.last_pos]
            self.nodelst[self.last_pos] = tmp
        self.last_pos -= 1
        check_bubble_down(self.nodelst, 0, self.last_pos)
```

下一个我们将实现合并排序。

```
**def** mergesort(array, i, l):
    **if** array == []:
        **return False
    if** len(array) == 2:
        **if** array[0] > array[1]:
            tmp = array[0]
            array[0] = array[1]
            array[1] = tmp
            **return** array
        **else**:
            **return** array
    **if** len(array) == 1:
        **return** array
    **else**:
        mid = (i + len(array))//2
        **return** merge(mergesort(array[i:mid], i, mid), mergesort(array[mid:l], i, mid+1))**def** merge(arr1, arr2):
    **if** (arr1 != **None**) **or** (arr2 != **None**):
        new_arr = []
        i = 0
        j = 0
        **while** ((i < len(arr1)) **and** (j < len(arr2))):
            **if** (arr1[i] > arr2[j]):
                new_arr.append(arr2[j])
                j += 1
                **if** j >= len(arr2):
                    **while**(i < len(arr1)):
                        new_arr.append(arr1[i])
                        i += 1
            **elif** arr1[i] <= arr2[j]:
                new_arr.append(arr1[i])
                i += 1
                **if** i >= len(arr1):
                    **while** (j < len(arr2)):
                        new_arr.append(arr2[j])
                        j += 1
        **return** new_arr
```

最后一个是快速排序。

```
**def** quick_sort(given, min, max):
    **if** given == **None or** len(given) == 0:
        **return False
    if** min > max:
        **return False** mid = min + (max-min)//2
    pivot = given[mid]
    i = min
    j= max
    **while** (i < j):
        **while** (given[i] < pivot):
            i += 1
        **while** (given[j] > pivot):
            j -= 1
        **if** (i <= j):
            tmp = given[j]
            given[j] = given[i]
            given[i] = tmp
            i += 1
            j -= 1
    **if** (min < j):
        quick_sort(given, min, j)
    **if** max > i:
        quick_sort(given, i, max)
    **return** given
```

我将引用这些算法背后的理论概念，在下面作为引用，在那里你可以得到算法运行时复杂性的解释。

参考资料:

1.  算法导论第三版[史密斯，朱迪斯 J，邓恩，芭芭拉 K]
2.  算法设计手册
3.  [斯坦福大学提供的算法专业](https://www.coursera.org/specializations/algorithms)【Tim rough garden】