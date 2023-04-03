# 用 3 个步骤编写 Python 嵌套列表理解

> 原文：<https://medium.com/analytics-vidhya/writing-python-nested-list-comprehensions-in-3-steps-75bd9a1b79b6?source=collection_archive---------11----------------------->

![](img/e2787506c7a0c16d1e3486d9bb1c215b.png)

照片由来自[派克斯](https://www.pexels.com/photo/close-up-photography-of-quail-eggs-on-nest-810320/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)的[伊兰尼特·科彭斯](https://www.pexels.com/@nietjuh?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)拍摄

Python 的可能性是无限的。对于你学到的每一个技巧和诀窍，似乎还有十几个可以吸收并用来让生活变得更容易。在我看来，作为一个 python 主义者，你能做的最巧妙的一件事就是列出理解。

对于那些不知道的人来说，列表理解是循环遍历迭代器(例如，像列表、元组或字典)的一行代码，并根据元素是否符合某些标准挑选出特定的项目。输出是符合条件的新答案列表。

例如:

```
first_ten = [1,2,3,4,5,6,7,8,9,10]evens = [number for number in first_ten if number % 2 == 0]evens = [2, 4, 6, 8, 10]
```

这是一个列表理解，它根据 first_ten 列表中的每个数字是否能被 2 整除来创建一个新的数字列表。

换句话说，我们告诉 python 从 first_ten 开始给我们一个条件为真的所有数字的新列表。另一种写法是:

```
evens = []for number in first_ten:
    if number % 2 == 0:
        evens.append(number)
```

列表理解的不同之处在于，你在*一行*中完成了完全相同的事情，而不是四行。感觉很棒，对吧？

# 嵌套列表理解

你可能想坐下来听接下来的部分。正如我们在上面问题的更长版本的“for”循环中嵌套了一个“if”条件一样，*你也可以在另一个*中嵌套一个列表理解。令人兴奋吧。

但是什么时候你需要一个嵌套列表理解呢？

假设您在字典中有一个包含姓名和电子邮件的地址簿，以及一个即将到来的聚会的单独邀请列表:

```
address_book =      {“Sally”: "sally@wahoo.com", 
      “John”: "john@froogle.com",
      “Veejay”: "veejay@botmail.com",  
      “Tom”: "tom@whymail.com", 
      “Meredith”: "meredith@schoolu.edu", 
      “Margot”: "margot@baeol.com}invite_list = ["John", "Veejay", "Margot"]
```

您想要遍历地址簿，为列表中的名字选择电子邮件地址，并创建一个新列表。这将很容易一次给每个人发电子邮件邀请参加聚会。

这是理解嵌套列表的绝佳机会:

```
invite_emails = [address_book[key] for name in invite_list for key in address_book if name in key]invite_emails = ["john@froogle.com", "veejay@botmail.com", "margot@baeol.com"]
```

# 3 步 Python 列表理解

所以我们遍历一个列表，对照字典交叉检查每个条目，我们也在遍历字典。嵌套循环方法可能有点难看，比如:

```
for item in invite_list:
    for key in address_book:
        if item == key:
            return value
```

但是这抛出了很多错误，总体上看起来很痛苦。不，不，我们应该用嵌套列表理解。我们可以做到这一点(答案见上文:我们已经做到了)！

当用 Python 构建嵌套列表理解时，就像软件开发中的许多事情一样，我喜欢从小处着手，一次做一件事。当我成功解决了问题的一小部分时，我就有足够的动力去解决未知的部分。

**步骤 1:** 在一个列表理解中，遍历 address_book 中的键，并用字典中的每个值的*填充新的列表理解。*

```
invite_emails = [address_book[key] for key in address_book]invite_emails = ["sally@wahoo.com", "john@froogle.com", "veejay@botmail.com", "tom@whymail.com", "meredith@schoolu.edu", 
"margot@baeol.com"]
```

成功！我们已经成功地遍历了字典，没有出现错误消息。这是很好的第一步，但是记住我们还需要遍历邀请列表。

**步骤 2:** 通过 invite_list 插入一个迭代。

```
invite_emails = [address_book[key] **for name in invite_list** for key in address_book]invite_emails = ['[sally@wahoo.com](mailto:sally@wahoo.com)', '[john@froogle.com](mailto:john@froogle.com)', '[veejay@botmail.com](mailto:veejay@botmail.com)', '[tom@whymail.com](mailto:tom@whymail.com)', '[meredith@schoolu.edu](mailto:meredith@schoolu.edu)', '[margot@baeol.com](mailto:margot@baeol.com)', '[sally@wahoo.com](mailto:sally@wahoo.com)', '[john@froogle.com](mailto:john@froogle.com)', '[veejay@botmail.com](mailto:veejay@botmail.com)', '[tom@whymail.com](mailto:tom@whymail.com)', '[meredith@schoolu.edu](mailto:meredith@schoolu.edu)', '[margot@baeol.com](mailto:margot@baeol.com)', '[sally@wahoo.com](mailto:sally@wahoo.com)', '[john@froogle.com](mailto:john@froogle.com)', '[veejay@botmail.com](mailto:veejay@botmail.com)', '[tom@whymail.com](mailto:tom@whymail.com)', '[meredith@schoolu.edu](mailto:meredith@schoolu.edu)', '[margot@baeol.com](mailto:margot@baeol.com)']
```

好吧，这有点疯狂。我们有三组电子邮件，而理想情况下我们只需要一组。这是因为我们要求 python 返回 invite_list 中项目的每个实例的键。三个项目意味着三个完整的邮件地址列表！

我们不希望这样，所以我们需要在这里设置一些限制…通过添加一个条件。

**第三步:**在嵌套列表理解的末尾添加条件。

```
invite_emails = [address_book[key] for name in invite_list for key in address_book **if name in key**]invite_emails = ['[john@froogle.com](mailto:john@froogle.com)', '[veejay@botmail.com](mailto:veejay@botmail.com)', '[margot@baeol.com](mailto:margot@baeol.com)']
```

…就是这样。一个新的列表，如果字典中的键碰巧在第二个列表中，则填充这些键。你这么说听起来很简单，对吧？(错误)

一般来说，嵌套列表理解(和常规列表理解)非常简洁，是 Python 的一个有用特性。使用它们已经成为我的第二天性。它们有助于将原本复杂的多行代码功能压缩到一行中。还有什么比这更好的呢？！？

如果您正在遍历两个不同的数据结构并比较信息，这将是部署这一独特代码(并给朋友留下深刻印象)的绝佳机会。但它有助于记住我最喜欢的一句关于生活和编程的名言:

> “计划不值钱，但规划就是一切。”

不要试图一次编写完整的嵌套 python 数据结构，也不要为了看看什么有效而把自己逼疯，对于如何编写 python 嵌套列表，一个更强的策略是制定一个计划。

使用上面的 3 步示例作为蓝图，如果幸运的话，你不会陷入测试每一种可能的方法来循环遍历你的两个迭代器以得到正确的答案。编码快乐！