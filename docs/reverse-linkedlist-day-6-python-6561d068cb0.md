# 反向链接列表—第 6 天(Python)

> 原文：<https://medium.com/analytics-vidhya/reverse-linkedlist-day-6-python-6561d068cb0?source=collection_archive---------4----------------------->

![](img/694e194146f8f3e8b0fa5d24f8436ef2.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Hush Naidoo](https://unsplash.com/@hush52?utm_source=medium&utm_medium=referral) 拍摄的照片

今天我们将学习如何反转链表。365 天挑战系列的第 2 天介绍了链表的基本特性。如果你需要复习，可以点击这个[链接](/@atharayil/palindrome-linkedlist-day-2-python-441a963697dc)。

[206](https://leetcode.com/problems/reverse-linked-list/)**。反向链表**

反转单向链表。

**示例:**

```
**Input:** 1->2->3->4->5->NULL
**Output:** 5->4->3->2->1->NULL
```

**跟进:**

链表可以迭代或递归地反转。你能两者都实现吗？

在我们开始解决问题之前，让我们再看几个例子。

```
**Input:** 1->NULL
**Output:** 1->NULL**Input:** 1->2->NULL
**Output:** 2->1->NULL**Input:** NULL
**Output:**NULL
```

解决方案—迭代方法

为了通过迭代方法反转一个链表，我们需要 3 个指针。

1.  保存上一个节点。
2.  保存当前节点。
3.  保存下一个节点。

我们只需要学习如何操作以上三个指针来反转一个链表。

1.  将先前的(A)指针初始化为“无”。
2.  让当前指针(B)指向链表的头部。
3.  运行循环，直到电流到达终点。
4.  下一个指针(C)指向当前的下一个。
5.  当前指针的下一个指向上一个。B -> A
6.  前一个指针现在是当前节点。
7.  当前指针现在是当前指针的下一个

```
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
         if head == None or head.next == None:
            return head
         prev, curr = None, head
         while curr:
               next_node = curr.next
               curr.next = prev
               prev = curr
               curr = next_node
         return prev
```

复杂性分析

**时间复杂度**

我们需要遍历 LinkedList 中的每个元素，这需要 O(N)时间。

**空间复杂度**

我们没有使用任何额外的数据结构来存储上述逻辑。因此空间复杂度是 O(1)。

解答——递归方法

当使用递归方法时，我们需要一个基本条件。

1.  对于这个问题，我们的基本条件是当 head 为 None 或 next of head 为 None。当我们到达基地时，我们需要返回。
2.  如果我们必须在到达基本条件之前运行几个节点，递归调用函数。
3.  头的下一个是头，头的下一个是无。
4.  然后返回电流。

```
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        if head == None or head.next == None:
            return head
        curr = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return curr
```

下面给出了上述逻辑的动画版本。

**时间复杂度**

我们需要遍历 LinkedList 中的每个元素，这需要 O(N)时间

**空间复杂度**

内部递归函数使用堆栈来执行程序。因此空间复杂度是 O(N)。

我想提高我的写作技巧，所以任何建议或批评都非常欢迎。