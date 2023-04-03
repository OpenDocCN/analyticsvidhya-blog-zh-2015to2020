# 两个夏天-第 4 天(Python)

> 原文：<https://medium.com/analytics-vidhya/two-sum-day-4-python-29ade633b634?source=collection_archive---------6----------------------->

![](img/9a667f32c6d509a51eb3a5ff9be214bf.png)

照片由克里斯·贾维斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

今天，我们将关注技术编码面试中面试官最喜欢的问题之一。

[**1**](https://leetcode.com/problems/two-sum/) **。两笔总和**

给定一个整数数组`nums`和一个整数`target`，返回这两个数的索引*，使它们相加为* `*target*`。

你可以假设每个输入都有 ***恰好*一个解决方案**，你不能两次使用*相同的*元素。

可以任意顺序返回答案。

**例 1:**

```
**Input:** nums = [2,7,11,15], target = 9
**Output:** [0,1]
**Output:** Because nums[0] + nums[1] == 9, we return [0, 1].
```

**例二:**

```
**Input:** nums = [3,2,4], target = 6
**Output:** [1,2]
```

**例 3:**

```
**Input:** nums = [3,3], target = 6
**Output:** [0,1]
```

**约束:**

*   `2 <= nums.length <= 105`
*   `-109 <= nums[i] <= 109`
*   `-109 <= target <= 109`
*   只有一个有效答案。

读完这个问题后，我想到的第一个解决方案是遍历每个元素，检查是否有任何两个元素相加得到答案。如果是，我们返回两个元素的索引，否则我们继续这个过程，直到找到答案。

代码如下所示。

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for a in range(len(nums)):
            for b in range(a+1, len(nums)):
                if nums[a]+nums[b] == target:
                    return [a,b]
```

复杂性

**时间复杂度**

考虑最坏的情况，其中我们的输出在列表的末尾。在这种情况下，我们将占用 O(N*(N-1))时间，即 O(N)。

**空间复杂度**

我们没有使用任何额外的空间，因此空间复杂度为 O(1)。

如何通过提高时间复杂度来让它变得更好？

我们可以使用一些额外的空间并跟踪已处理的数字吗？

我们需要找到两个数字，它们的总和等于目标值。

> a + b =目标

如果我们把问题修改如下呢？

> b =目标-a

我们可以使用一个字典来存储已经处理过的数字，并检查字典中是否有当前数字和目标的减法。如果是，则返回索引，否则将当前数字与其索引一起放入字典。

让我们看看上面逻辑的代码。

```
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        processed_number_dict = dict()
        for a in range(len(nums)):
            if (target-nums[a]) in processed_number_dict.keys():
                return([a, processed_number_dict[target-nums[a]]])
            else:
                processed_number_dict[nums[a]] = a
```

复杂。

**时间复杂度**

我们遍历数组一次，因此时间复杂度为 O(N)

**空间复杂度**

我们使用字典来存储数组中的元素。在最坏的情况下，如果我们的两个数字都在数组的末尾，那么我们将在字典中存储 N-2 个数字。因此上述解决方案的空间复杂度是 O(N)。

我想提高我的写作技巧，所以任何建议或批评都非常欢迎。