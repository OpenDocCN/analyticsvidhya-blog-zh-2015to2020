# 面试问题:邻居的灯

> 原文：<https://medium.com/analytics-vidhya/interview-question-neighbors-lights-7b7940dd36f6?source=collection_archive---------27----------------------->

![](img/a243cc263b3d50176e81a0e777aef15a.png)

软件工程师的技术面试因晦涩难懂的问题而臭名昭著。面试官的目标是观察你如何处理问题，并试图理解你的思维过程。当你想出解决问题的正确方法时，你越能把你的思考过程用语言表达出来，你就越有可能获得后续面试的机会。

这是我最近在一家硅谷公司面试时遇到的一个问题。事情是这样的:

> 想象一条街上有一排八栋房子。每栋房子的前廊都有一盏灯，主人会在一天开始时打开或关闭。午夜时分，业主们走出家门，决定在接下来的 24 小时里，他们的灯是亮着还是关着。为了决定灯是开还是关，主人会查看两个邻居家的灯。如果两个邻居的灯不是开着就是关着，主人就会打开他的灯。否则，灯将被关闭。由于街道远端的两栋房子只有一个邻居，所以只有在他们邻居的灯前一天关了的情况下，他们才会开灯。
> 
> 写一个函数，取未来的一天，n，和一个表示第 0 天家庭灯光状态的数组，返回第 n 天的灯光状态。

乍一看，这听起来有点令人生畏。所有这些谈论灯光和邻居等等。但是，让我们花点时间来分析一下。为了简化问题，让我们忽略整个“第 n 天的灯光状态”的事情，让我们弄清楚如何根据今天的状态来确定明天的灯光。此外，让我们弄清楚如何最好地模拟这个问题。

所以，我们知道我们需要一个数组来表示某一天的灯光状态。由于每盏灯要么只能是*开*要么是*关*，所以我们只使用 1 和 0 的数组来表示各自的状态是有意义的。因此，我们的输入数组可能看起来像这样:

```
state = [0, 1, 0, 1, 0, 1, 1, 0]
```

在这个场景中，灯是*关，开，关，开，关，开，开，关。*直观上，我们可以看到这一点，并准确地看到第二天应该打开哪些灯。但是我们如何用代码做到这一点呢？

天真的方法是进行某种循环并确定灯 *i* 是否应该打开，查看*I-1*和 *i + 1* 。这种方法可行，但我们需要对第*0*和第*7*位置进行特殊处理，因为前者没有*I-1*，后者也没有 *i + 1* 邻居。该代码可能类似于以下内容:

```
tomorrows_lights = [0, 0, 0, 0, 0, 0, 0, 0] # Handle special cases. Only on if neighbor is off tomorrows_lights[0] = int(not state[1]) 
tomorrows_lights[1] = int(not state[6]) i = 1 
while i < 8: 
    tomorrows_lights[i] = int(state[i - 1] == state[i + 1]) 
    i += 1
```

这还不算太糟糕。我利用 Python falsy 值在适当的位置填充 1 或 0。虽然这是可行的，但我认为还有更好的解决方案。

因为我们在这里处理 1 和 0，所以使用按位运算符是一个可以探索的选项。因此，让我们将输入数组转换成按位表示。

```
lights = functools.reduce(lambda a, b: (a << 1) | b, state)
```

这一行有效地将输入数组转换成 8 位整数。对于状态数组中的每一项，值都向左移动一个位置，并从状态数组中追加 0 或 1。所以，我们的输入数组`[0, 1, 0, 1, 0, 1, 1, 0]`变成了`86`或者`01010110`。

现在，我们可以使用按位运算符来确定两个邻居是否都很容易。如果我们执行一次左移和一次右移，然后对这两次移位执行按位“与”运算，只有在两个邻居都亮着灯的情况下，1 才会出现。举个例子，

```
left_shift = lights << 1                    #10101100 
right_shift = lights >> 1\.                  #00101011 
tomorrows_lights = left_shift & right_shift #00101000
```

这处理了两个邻居在上处于初始状态*的情况。但是我们如何确定两个邻居是否都是 *OFF* ？实际上很容易。如果两个邻居都*关断*，那么左移位和右移位对于相同的位都将具有 0。因此，让我们对两者进行按位“或”运算，看看会产生什么结果:*

```
left_shift | right_shift # 10101111
```

注意应该在上*出现的灯在两个有零的地方。为了得到正确的状态，我们可以用掩码`0xFF`执行 XOR 运算*

```
(left_shift | right_shift) ^ 0xFF # 01010000
```

通过将此与前面的 AND 运算相结合，我们获得了明日之光的解决方案。我们可以将这些操作合并到一行中，使代码更加简洁。

```
left_shift = lights << 1   #10101100 
right_shift = lights >> 1  #00101011 
tomorrows_lights = (left_shift & right_shift) | ((left_shift | right_shift) ^ 0xFF) & 0xFF
```

注意，为了确保只设置底部的 8 位，我们用`0xFF`执行了一个 AND。

好了，这三行代码完成了所有繁重的工作。为了概括这一点，并定义一个函数，可以找到未来任意一天的光序列，我们将把这个位放在一个循环中，并为每一天执行这个位操作。

```
def get_light_sequence_on_day_n(n, state): 
    lights = functools.reduce(lambda a, b: (a << 1) | b, state) 
    while n > 0: 
        left_shift = lights << 1 
        right_shift = lights >> 1 
        lights = (left_shift & right_shift) | ((left_shift | right_shift) ^ 0xFF) & 0xFF 
        n -= 1 return lights
```

我们的函数即将完成，但是现在的返回值是一个数字，并且需要一个表示灯光状态的数组。为此，我们需要在该函数的第一行创建 reduce 语句的反向操作。我想到的解决方案是首先使用 Python 的 string format 方法将返回值格式化为二进制字符串，使用 map 遍历该字符串，然后返回结果列表。看起来像这样:

```
list(map(lambda n: int(n), "{0:08b}".format(light_sequence)))
```

因此，完成的函数看起来只有 8 行长，相当容易理解，并且比使用 for 循环和处理边界情况更优雅。

```
def get_light_sequence_on_day_n(n, state): 
    lights = functools.reduce(lambda a, b: (a << 1) | b, state) 
    while n > 0: 
        left_shift = lights << 1 
        right_shift = lights >> 1
        lights = (left_shift & right_shift) | ((left_shift | right_shift) ^ 0xFF) & 0xFF 
        n -= 1 
    return list(map(lambda n: int(n), "{0:08b}".format(light_sequence)))
```

查看我的 [GitHub](https://github.com/jcampos8782/codingz/blob/master/py/lighting_sequence.py) 上的完整源代码。如果你有任何建议，请在评论中留下。感谢阅读，敬请期待更多内容。

*原载于 2020 年 2 月 24 日 http://blog.jsoncampos.com*[](http://blog.jsoncampos.com/2020/02/24/interview-question-which-lights/)**。**