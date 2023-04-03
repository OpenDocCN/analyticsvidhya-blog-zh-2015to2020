# 在 Python 中重温雨滴

> 原文：<https://medium.com/analytics-vidhya/revisiting-raindrops-in-python-abc4fbb23f4e?source=collection_archive---------15----------------------->

![](img/a36065f6294a6260c07967aaacfe9d75.png)

不久前，我在[exercisem . io](http://exercism.io)上的 Python 轨道中展示了[对雨滴问题](/@losimprov/raindrops-in-python-e72a3e4941ab)的解决方案——类似于经典的 FizzBuzz 问题。

下面是这个问题的简要回顾:

```
Your task is to convert a number into a string that contains raindrop sounds corresponding to certain potential factors. A factor is a number that evenly divides into another number, leaving no remainder. The simplest way to test if a one number is a factor of another is to use the [modulo operation](https://en.wikipedia.org/wiki/Modulo_operation).The rules of raindrops are that if a given number:
• has 3 as a factor, add ‘Pling’ to the result.
• has 5 as a factor, add ‘Plang’ to the result.
• has 7 as a factor, add ‘Plong’ to the result.
*• does not* have any of 3, 5, or 7 as a factor, the result should be the digits of the number.**Examples**28 has 7 as a factor, but not 3 or 5, so the result would be “Plong”.30 has both 3 and 5 as factors, but not 7, so the result would be “PlingPlang”.34 is not factored by 3, 5, or 7, so the result would be “34”.
```

解决方案是:

```
def convert(number):
 output = ‘’
 if number % 3 == 0:
 output += ‘Pling’if number % 5 == 0:
 output += ‘Plang’if number % 7 == 0:
 output += ‘Plong’if output is ‘’:
 return str(number)

 return output
```

没事的。这是一个非常实用、有效的解决方案。一匹老黄牛。但这并不能完全扩大规模。如果我们想最终增加几十个因素呢？我们必须为每一个都写“如果”语句吗？将这些因素放在一个数据结构中，然后遍历它会更有效。此外，用 Python 创建字符串相对较慢，因为程序会在运行时不断地重新创建然后销毁这些数据。将单个字符串附加到单个列表中，然后连接该列表，如果我们最终要扩大规模，会使程序运行得更快。

考虑到这些因素，让我们试着重构这个坏男孩。

一些需要记住的事情:

*   这种解决方案在许多因素下都不太适用。将这些因素放入数据结构并遍历它会更有效。
*   在 Python 中创建字符串相对较慢。最好不要经常这样做，尤其是对于较大的程序。
*   可以在 returns 语句中包含 or 条件。你可以写`if not output`，或者更短的`return output or str(number)`。

考虑到这一点，我们得到这个:

```
def convert(number):
    factors = []
    output = ''
    tu = {1:'Pling', 2:'Plang', 3:'Plong'}

    if number % 3 == 0:
        factors += tu[1]

    if number % 5 == 0:
        factors += tu[2]

    if number % 7 == 0:
        factors += tu[3]

    return output.join(factors) or str(number)
```

通过使用字典，我们可以以多种方式迭代和评估*号*。如果*号*满足要求，我们在那个特定的键上追加字典的值。

如果我们想给这个程序添加更多的需求，我们可以把它们添加到字典中，并添加新的‘if’子句。

它变短了吗？没有。它是否通过不连续创建字符串来遵循经济数据使用的最佳实践？你打赌。它是否建立了一个数据结构，使得迭代和评估变得简单？毋庸置疑。

我们不是在每次运行程序时都创建字符串，而是将单个字符串值附加到一个列表中，以后再将它们连接起来。

请继续关注《雨滴》的激动人心的结局，当我们拿出切肉刀，开始对那些“如果”陈述做一些修整…