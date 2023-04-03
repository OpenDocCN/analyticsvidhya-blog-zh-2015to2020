# 时间复杂性:编写更快 Python 程序的(不那么)隐藏的秘密

> 原文：<https://medium.com/analytics-vidhya/time-complexity-the-not-so-hidden-secret-to-writing-faster-python-programs-6200a2872d6a?source=collection_archive---------18----------------------->

![](img/393de8815a05c8d7e07fb8fd52ca14d6.png)

照片由来自[佩克斯](https://www.pexels.com/photo/blue-tree-painting-2397989/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)的[安德烈·布伦南](https://www.pexels.com/@andree-brennan-974943?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)拍摄

作为一名初学 web 开发的人，启动并运行一个程序是一件令人兴奋的事情。有很多次，当我遇到一个特别令人沮丧的工作问题时，我会对着空气(不针对任何人)挥舞拳头。

但是，随着您在旅程中的进一步发展，您可能需要开始考虑扩展程序的复杂性和挑战。

> 随着你成为一名程序员，你开始较少考虑你将如何解决一个特定的问题，而是更多地考虑你为什么以这样或那样的方式解决它。

好吧…我撒谎了。你仍然考虑如何做，但是你也必须考虑为什么做。

当然，你的矩阵程序可以处理一个 3 x 3 的正方形，但是如果你把它扩大到 9 x 9，甚至 90 x 90 呢？你写的表达式是否足够精简以承受更大规模的执行，或者它们是否会降低你的运行时间，或者更糟，导致你的整个程序崩溃？你不会想等到生产时才发现这一点。最好在第一次编写精益和吝啬的代码，并谨慎行事。

最近，我在为[exercisem . io](http://Exercism.io)做的一个问题中遇到了这类问题。这个问题包括给特定的字母分配不同的分值，并报告任何给定单词的分数，就像你在拼字游戏中所做的那样:

```
Letter                                Value
A, E, I, O, U, L, N, R, S, T            1D, G                                    2B, C, M, P                              3F, H, V, W, Y                           4K                                       5 J, X                                    8Q, Z                                    10
```

在程序结束时，您需要返回给定单词的完整分数。例如，单词“PYTHON”的得分为 14 (3+4+1+4+1+1)。

对于这个问题，我最初的解决方案是将上述所有值放入一个字典中。上表中“字母”列的每一行都是它自己的键，如下所示:

```
def score(word):

     letter_values = { "AEIOULNRST": 1,
        "DG": 2,
        "BCMP": 3,
        "FHVWY": 4,
        "K": 5,
        "JX": 8,
        "QZ": 10

         }

     score = [letter_values[key] for letter in list(word.upper())      
     for key in letter_values if letter in key]

     return sum(score)
```

在这个解决方案中，*分数*是一个列表。更准确地说，这是一种嵌套列表理解，它遍历 *letter_values* 中的每个键以及 *word* 中的每个字母，并在 *word* 中追加与给定字母匹配的键的值。

然后，该函数返回该分数列表中所有值的总和，以得到一个完全可用的解决方案。

如果我们从未计划扩大规模。

你看，上面的解决方案中部署的漂亮的列表理解相当复杂。也就是说，它要求计算机做大量的迭代来得到正确的解。在这里，它被解开了:

```
for letter in word:
    letter_score = letter_values[letter] score.append(letter_score)
```

这很好，但是如果这个单词有 100 万个字母呢？(我不认为任何语言有那么长的单词，甚至德语也没有，只是跟着它走)。这将需要大量的迭代，因为计算机会在我们的百万字母单词中逐个字母地进行迭代，然后遍历上面的每个键来定位正确的字母，然后插入该字母的相应值。光是想想就让我精疲力尽。

这里的概念是时间复杂性。[你可以在这里阅读更多关于时间复杂度的内容](https://www.techopedia.com/definition/22573/time-complexity)。

> 时间复杂度是算法处理或运行所需时间的量化，是输入量的函数。基本上，它是对你的程序在运行时效率的度量。

老实说，我不完全确定这个程序运行一百万个字母的单词会不会慢一些，我从来没有试过。但这是一个可以重构的低效程序。这是一个很好的机会来探索这个概念，并确保我们编写的代码能够预见到即将出现的这类问题。毕竟，一盎司的预防抵得上一磅的治疗，正如一位[的印刷工人](https://en.wikipedia.org/wiki/Benjamin_Franklin)曾经说过的。(顺便说一句，显然他说这话时指的是防火？[感谢互联网。](https://www.ag.ndsu.edu/news/columns/beeftalk/beeftalk-an-ounce-of-prevention-is-worth-a-pound-of-cure/))

因此，百万美元(或百万字母)的问题是:如何重构它，使程序更有效？

首先，我们可以使用更多的代码行。虽然这看起来可能违反直觉，但通过将键分解成每个字母并拥有 26 个不同的键，我们实际上帮助程序运行得更快。这是我们在计算机执行的任务列表中要担心的迭代的一个小步骤。

然后，我们可以简单地在给定的键上附加字母的值，就像这样:

```
def score(word):

    letter_values = {
        'A': 1,
        'E': 1,
        'I': 1,
        'O': 1,
        'U': 1,
        'L': 1,
        'N': 1,
        'R': 1,
        'S': 1,
        'T': 1,
        'D': 2,
        'G': 2,
        'B': 3,
        'C': 3,
        'M': 3,
        'P': 3,
        'F': 4,
        'H': 4,
        'V': 4,
        'W': 4,
        'Y': 4,
        'K': 5,
        'J': 8,
        'X': 8,
        'Q': 10,
        'Z': 10
    }
    score = [letter_values[letter] for letter in word.upper()]
    return sum(score)
```

该程序通过将*letter _ value【letter】*插入分数列表，返回每个字母的正确值。

我们没有遍历*单词*和 *letter_values* 字典的键，而是遍历*单词*中的字母，并将它们直接插入字典中以获取值。

这是一个更简单、更优雅的解决方案。我猜，如果我们的计算机会说话，它们会感谢我们减少了它们生活中的时间复杂性。