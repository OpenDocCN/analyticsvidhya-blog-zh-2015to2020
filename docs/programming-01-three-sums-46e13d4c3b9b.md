# 编程 01-三个和

> 原文：<https://medium.com/analytics-vidhya/programming-01-three-sums-46e13d4c3b9b?source=collection_archive---------22----------------------->

让我们深入一个惊人的编程问题，以提高您解决问题的技能。

我猜你是:)

问题是-

给定一个由 *n* 个整数组成的数组`nums`，在`nums`中是否有元素 *a* 、 *b* 、 *c* 使得*a*+*b*+*c*= 0？找出数组中所有唯一的三元组，其和为零。

请注意，解决方案集不得包含重复的三元组。

**例 1:**

```
**Input:** nums = [-1,0,1,2,-1,-4]
**Output:** [[-1,-1,2],[-1,0,1]]
```

**例二:**

```
**Input:** nums = []
**Output:** []
```

**例 3:**

```
**Input:** nums = [0]
**Output:** []
```

**约束:**

*   `0 <= nums.length <= 3000`
*   `-100000 <= nums[i] <= 100000`

在你直接进入解决方案之前，仔细观察这个问题。

如果您错过了这个问题，您可能会错过正确答案-“数组中唯一的三个一组”。

在直接编码之前，我更喜欢通过写下可能的解决方法来解决问题，当然也可以试运行。

1.  让我们先对数组进行排序。
2.  排序后，我们可以保持向前和向后两个指针，向前从索引 i+1 开始，I 从 0 开始，向后从数组的末尾开始。
3.  然后，只有当所有三个元素都是唯一的，我们才能找到第 I 个索引、前向元素和后向元素的总和，并检查总和是否等于零。
4.  如果总数等于零，则在答案后面加上三元组，然后向前递增，向后递减。
5.  如果总数为负，则向前递增，如果总数为正，则向后递减。
6.  在通过条件前进<backward increment="" i="" and="" repeat="" the="" process=""></backward>

```
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:

        ans=[]
        nums.sort()

        length=len(nums)

        for i in range(length-2):
            if (i>0 and nums[i]==nums[i-1]):
                continue
            l=i+1;
            r=length-1

            while(l<r):
                total=nums[i]+nums[l]+nums[r];
                if(total<0):
                    l=l+1
                elif total>0:
                    r=r-1
                else:
                    ans.append([nums[i],nums[l],nums[r]])
                    while l<r and nums[l]==nums[l+1]:
                        l=l+1
                    while l<r and nums[r]==nums[r-1]:
                        r=r-1
                    l=l+1
                    r=r-1
        return ans
```

The Time Complexity is O(n log n)+O(n²)=O(n²). Note here O(n log n) is used for the**遍历整个长度后，排序功能**和 O(n)用于**两个循环**。由于 O(n)比 O(n log n)更占优势，所以时间复杂度变成了 **O(n )** 。

由于我们创建了一个名为 **ans** 的链表，所以辅助空间复杂度为 **O(n)** 。

编码快乐！！！