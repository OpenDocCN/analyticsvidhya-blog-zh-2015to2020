# [Leet 代码]最多 69 个数字

> 原文：<https://medium.com/analytics-vidhya/leet-code-maximum-69-number-c24f24bd5890?source=collection_archive---------11----------------------->

leet code:[https://leetcode.com/problems/maximum-69-number/](https://leetcode.com/problems/maximum-69-number/)

问题:

```
Given a positive integer num consisting only of digits 6 and 9.Return the maximum number you can get by changing **at most** one digit (6 becomes 9, and 9 becomes 6).
```

示例:

```
**Input:** num = 9669
**Output:** 9969
**Explanation:** 
Changing the first digit results in 6669.
Changing the second digit results in 9969.
Changing the third digit results in 9699.
Changing the fourth digit results in 9666\. 
The maximum number is 9969.
```

解决方案:

```
class Solution(object):
    def maximum69Number (self, num):
        """
        :type num: int
        :rtype: int
        """

        changed_num = False
        nums = list(str(num))

        for i in range(len(nums)):
            print nums[i]
            if changed_num == False:
                if nums[i] == "6":
                    nums[i] = "9"
                    changed_num = True
            else:
                continue

        a_string = "".join(nums)
        an_integer = int(a_string)
        return an_integer
```

解释:

我们必须从一开始就意识到，最大数量将通过改变第一个最左边的 6 来实现。

因此，如果我们首先创建一个名为“changed_num”的变量，它会告诉我们是否已经更改了其中一个值。然后我们简单地遍历已经转换成字符串的数字列表，用 9 替换前 6。然后我们把它转换回一个整型数，并返回这个整型数。

![](img/1a5106235eaac78c953017f434f9c4b0.png)