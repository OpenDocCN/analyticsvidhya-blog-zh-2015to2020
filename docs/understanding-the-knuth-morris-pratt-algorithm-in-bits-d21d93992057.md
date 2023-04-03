# 理解 Knuth Morris Pratt 算法…以比特为单位

> 原文：<https://medium.com/analytics-vidhya/understanding-the-knuth-morris-pratt-algorithm-in-bits-d21d93992057?source=collection_archive---------7----------------------->

![](img/23880fb675b11db159b6a217eb806b2f.png)

格德·奥特曼([https://pixabay.com/users/geralt-9301/](https://pixabay.com/users/geralt-9301/))拍摄的照片

字符串匹配是计算机科学中的一项重要任务，具有从搜索数据库到遗传学的广泛应用。有许多算法可以完成这项任务，在本文中，我将带您了解 Knuth Morris Pratt 算法 **(KMP)** 。

# 天真的方式

在我们进入 **KMP** 算法之前，我想展示一下大多数人是如何尝试解决字符串匹配问题的。我这样做是为了让我们可以看到 KMP 是如何优化搜索的。

最简单但效率最低的搜索匹配的方法是遍历**干草堆**中的每个字符，并将该字符及其后的字符与**针**中的字符进行比较。下面的代码片段展示了我们如何做到这一点。

虽然我们可以通过在内循环中发现字符不匹配时立即将外循环推进一个字符来优化上面的代码片段，但是，当我们在最后一个字符之前都有匹配的字符时，这将给我们带来最坏情况下的时间复杂度 **O(nm)** 。

```
haystack = AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAneedle = AAAAAAAAB
```

# KMP 之路

**KMP** 算法从左至右遍历**干草堆**和**针**直到找到匹配。下面是两个字符串长度相等的特殊情况下的代码片段。这是为了让你习惯于 **KMP** 如何遍历字符串来寻找匹配。

## 移动指针

既然我们看到了 **KMP** 如何确定一个字符是否匹配，我们接下来将看到 **KMP** 如何在【T30(n)】时间内移动**指针**通过**干草堆**来搜索匹配。看看下面的代码片段。

*注意:这里提供的算法只适用于在这之后的代码片段中给出的测试用例，并将跳过其他测试用例中可能的匹配。这样做只是为了展示如何将针穿过干草堆。*

每当发现不匹配并且其索引大于 0 时，我们立即将**指针**移动到**干草堆**中的那个字符，并开始我们的比较。如果不匹配发生在**针**的索引 0 处，我们将**针**移动到不匹配后的下一个字符。这有一个最好和最坏的时间复杂度 **O(n)** 。

```
# HOW KMP MOVES THE NEEDLE THROUGH THE HAYSTACK**C**ABCDEKLSDSABCDEABCD
**A**BCDEABCDCABCDE**K**LSDSABCDEABCD
 ABCDE**A**BCDCABCDE**K**LABCDEABCD
      **A**BCDEABCDCABCDEK**L**ABCDEABCD
       **A**BCDEABCDCABCDEKL**ABCDEABCD**
        **ABCDEABCD**
```

## **前缀和后缀**

尽管上面的算法给出了相当好的时间复杂度 **O(n)** ，但是它会跳过一些可能的匹配，仍然会进行不必要的比较，因此不会给出最佳的效率。为了进一步提高上述字符串匹配算法的效率， **KMP** 使用了**针**中包含的适当的前缀和后缀来避免不必要的比较。

给定一个字符串 **ABCDEFABCR** ，我们在这个字符串中的正确前缀是 **A，AB，ABC，ABCD，ABCDE，ABCDEF，ABCDEFA，ABCDEFAB，ABCDEFABC** ，后缀是 **R，CR，BCR，ABCR，FABCR，EFABCR，DEFABCR，CDEFABCR，BCDEFABCR** 。现在，我们可以看到 **A、AB、ABC** 是正确的前缀，它们也可以在字符串 **ABCDEFABCR** 中包含的后缀中找到。

在 **KMP** 中，我们构建了一个数组，告诉我们这些正确的前缀(也是**指针**的后缀)出现在哪里，我们用它来优化我们的搜索。下面的代码片段构建了一个数组，告诉我们**“最长的正确前缀也是一个后缀(LPS)”**出现在哪里。

## 使用 LPS 阵列

当我们发现正在比较的字符不匹配，并且该字符的索引不是第一个索引(0)时，我们查找该字符之前的 **LPS** 值，以了解在**针**中可以跳过多少个字符。然后，我们开始将不匹配的字符与最后跳过的字符之后的字符进行比较。如果第一个字符出现不匹配，我们将**针**移动到**干草堆**中的下一个字符。

```
# HOW KMP MOVES UTILIZES THE LPS
The LPS for **ABCDEABCD = [0,0,0,0,0,1,2,3,4]****C**ABCDEKLSDSABCDEABCD
**A**BCDEABCDCABCDEAB**R**DABCDEABCD
 ABCDEAB**C**D# **A** and **B** are italicized to show that they are skipped. 
# Rather than begin comparing from index 0, we start our comparison from index **2
#** The **possible match** ***AB*** ishowever not skipped even though we never compare the two characters.CABCDEAB**R**DABCDEABCD
      *AB***C**DEABCDCABCDEABR**D**ABCDEABCD
         **A**BCDEABCDCABCDEABRD**ABCDEABCD**
          **ABCDEABCD**
```

完整的 Knuth Morris Pratt 算法