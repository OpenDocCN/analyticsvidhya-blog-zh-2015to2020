# 滑动窗口算法

> 原文：<https://medium.com/analytics-vidhya/sliding-window-algorithm-ed04be066922?source=collection_archive---------29----------------------->

在这篇文章中，我将整理我遇到的所有与滑动窗口算法相关的问题，从初级到高级。这些是我在实习时发现的一些笔记，我希望它们能帮助你将基础知识提升到高级水平。关于这个算法的入门文章，可以去翻一下[这个](https://www.geeksforgeeks.org/window-sliding-technique/)。

![](img/52a3a595839cd1407519f49392e34ce0.png)

# 没有重复字符的最长子字符串

## 问题:

给定一个字符串，找出没有重复字符的**最长子串**的长度。

## 示例:

**输入**:“abc ABC bb”
**输出** : 3
**解释**:答案是“ABC”，长度为 3。

## 方法:

使用 HashSet 存储当前窗口[i，j]中的字符。然后我们向右滑动索引 *j* 。如果它不在哈希表中，我们进一步滑动 *j* 。这样做，直到 s[j]已经在 HashSet 中。此时，我们发现从索引 *i* 开始的没有重复字符的子字符串的最大大小。如果我们对所有人都这样做，我们就会得到答案。

## 浏览:

```
abcabcbb
i
j 
set = [a], ans = max(0, 0-0+1) = 1abcabcbb
ij 
set = [a, b], ans = max(1, 1-0+1) = 2abcabcbb
i j 
set = [a, b, c], ans = max(2, 2-0+1) = 3abcabcbb
i  j 
repeating character found at j, so i++ and remove char at i from set
set = [b, c]abcabcbb
 i j 
set = [b, c, a], ans = max(3, 3-1+1) = 3abcabcbb
 i  j 
repeating character found at j, so i++ and remove char at i from set
set = [c, a]abcabcbb
  i j 
set = [c, a, b], ans = max(3, 4-2+1) = 3abcabcbb
  i  j
repeating character found at j, so i++ and remove char at i from set
set = [a, b]abcabcbb
   i j 
set = [a, b, c], ans = max(3, 5-3+1) = 3abcabcbb
   i  j 
repeating character found at j, so i++ and remove char at i from set
set = [b, c]abcabcbb
     ij
repeating character found at j, so i++ and remove char at i from set
set = [c]abcabcbb
     ij
set = [c, b], ans = max(3, 6-5+1) = 3abcabcbb
     i j
repeating character found at j, so i++ and remove char at i from set
set = [b]abcabcbb
      ij
repeating character found at j, so i++ and remove char at i from set
set = []
```

## 代码:

没有重复字符的最长子字符串

# 最多包含两个不同字符的最长子字符串

## 问题:

给定一个字符串 S，找出最多包含两个不同字符的最长子字符串 T 的长度。

## 示例:

**输入**:【aabcd】
**输出** : 3
**解释**:答案是“aab”，长度为 3。

## 方法:

我们使用一个滑动窗口，它总是满足这样一个条件，其中最多总是有两个不同的字符。当我们添加一个打破这个条件的新字符时，我们移动字符串的起始指针。

## 浏览:

```
aabcd
i
j
k = 0 (character 'a')
ans = max(0, 0-0+1) = 1aabcd
ij
k = 1 (characte 'a')
ans = max(1, 1-0+1) = 2aabcd
i j
k = 2 (character 'b')
ans = max(2, 2-0+1) = 3aabcd
i  j
more than two distict characters found, i++aabcd
 i j
more than two distict characters found, i++aabcd
  ij
ans = max(3, 3-2+1) = 3aabcd
  i j
more than two distict characters found, i++aabcd
   ij
ans = max(3, 4-3+1) = 3
```

## 代码:

最多包含两个不同字符的最长子字符串

# 最多包含 K 个不同字符的最长子字符串

## 问题:

对前面问题的扩展，但是现在子串中需要 k 个不同的字符，而不是 2 个。

## 示例:

**输入** : s = "eceba "，k = 2
**输出** : 3
**解释**:答案是“ece”，长度为 3。

## 方法:

这类似于用最多两个不同的字符来解决前面的问题，但是现在唯一的不同是我们还需要跟踪不同字符的数量，为此我们使用了地图的帮助。

最多包含 K 个不同字符的最长子字符串

# 具有精确的 K 个不同字符的最长子字符串

## 问题:

给定一个正整数数组 A，找出正好有 K 个不同字符的子数组的个数。

## 示例:

**输入**:A =【1，2，1，2，3】，K = 2
**输出** : 7
**说明**:恰好由两个不同的整数组成的子阵列:[1，2]，[2，1]，[1，2]，[2，3]，[1，2，1]，[2，1，2]，[1，2，1，2]，[1，2，1，2]。

## 方法 1(智能工作):

如果我们知道如何找到具有“最多 k 个不同字符”的子阵列，那么我们可以使用以下等式扩展上述算法来找到具有“正好 k 个不同字符”的子阵列的数量:

```
exactly(K) = atMost(K) — atMost(K-1)`
```

## 方法 1 的代码:

具有精确的 K 个不同字符的最长子字符串

## 方法 2(努力工作):

[](https://leetcode.com/problems/subarrays-with-k-different-integers/discuss/235235/C%2B%2BJava-with-picture-prefixed-sliding-window) [## C++/Java 带图片，带前缀的滑动窗口- LeetCode 讨论

### 如果问题涉及到连续的子阵列或子串，滑动窗口技术可能有助于用一种简单的方法解决它

leetcode.com](https://leetcode.com/problems/subarrays-with-k-different-integers/discuss/235235/C%2B%2BJava-with-picture-prefixed-sliding-window) 

> 感谢您的阅读！如果您觉得这很有帮助，以下是您可以采取的一些后续步骤:

1.  给我点掌声！
2.  在[媒体](/@dimi.soty_61150)上关注我，在 [LinkedIn](https://www.linkedin.com/in/deeheem-ansari-902aa6147/) 上联系我！