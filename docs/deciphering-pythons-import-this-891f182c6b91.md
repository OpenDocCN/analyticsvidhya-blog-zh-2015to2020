# 解密 Python 的“导入这个”

> 原文：<https://medium.com/analytics-vidhya/deciphering-pythons-import-this-891f182c6b91?source=collection_archive---------14----------------------->

![](img/097f45f0efa0be3268464a06cf7ed9d5.png)

照片由 [**Pixabay**](https://www.pexels.com/@pixabay?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 来自 [**像素**](https://www.pexels.com/photo/background-balance-beach-boulder-289586/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

在众所周知的复活节彩蛋中，Python 嵌入了一个库，用于显示其著名的 PEP 20，称为 Python 的*Zen*，其中包含 19 条格言，为 Python 程序员提供指导。

# Python 的印刷禅

只需导入`this`就可以打印出这 19 句格言。

```
import this
```

这输出

```
The Zen of Python, by Tim PetersBeautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

# 解读 Python 的禅

`this`其实还有更多。该模块提供了四个变量:`this.c`、`this.d`、`this.i`和`this.s`。`this.s`包含 Python 的 *Zen 的加密版本，使用字母表的排列，可以在`this.d`字典中恢复。`print(this.s)`慨然*

```
Gur Mra bs Clguba, ol Gvz CrgrefOrnhgvshy vf orggre guna htyl.
Rkcyvpvg vf orggre guna vzcyvpvg.
Fvzcyr vf orggre guna pbzcyrk.
Pbzcyrk vf orggre guna pbzcyvpngrq.
Syng vf orggre guna arfgrq.
Fcnefr vf orggre guna qrafr.
Ernqnovyvgl pbhagf.
Fcrpvny pnfrf nera'g fcrpvny rabhtu gb oernx gur ehyrf.
Nygubhtu cenpgvpnyvgl orngf chevgl.
Reebef fubhyq arire cnff fvyragyl.
Hayrff rkcyvpvgyl fvyraprq.
Va gur snpr bs nzovthvgl, ershfr gur grzcgngvba gb thrff.
Gurer fubhyq or bar-- naq cersrenoyl bayl bar --boivbhf jnl gb qb vg.
Nygubhtu gung jnl znl abg or boivbhf ng svefg hayrff lbh'er Qhgpu.
Abj vf orggre guna arire.
Nygubhtu arire vf bsgra orggre guna *evtug* abj.
Vs gur vzcyrzragngvba vf uneq gb rkcynva, vg'f n onq vqrn.
Vs gur vzcyrzragngvba vf rnfl gb rkcynva, vg znl or n tbbq vqrn.
Anzrfcnprf ner bar ubaxvat terng vqrn -- yrg'f qb zber bs gubfr!
```

让我们来破译它

```
import this
deciphered = ""
for c in this.s:
    try:
        decrypted += this.d[c]
    except:                     # If the symbol is not encrypted
        decrypted += c
print(deciphered)
```

# 结论

这就是这个有趣的复活节彩蛋！作为临别赠言，要理解 Python 的*禅*如何嵌入 Python 的哲学，考虑一下“实用性战胜纯粹性”。该语句在`len`函数的设计中形成，这是特殊方法中的一个例外，因为出于实用的原因，当实现自己的类时，它可以这样使用，但为了速度，当用于基本数据结构(如列表、集合和字典)时，它被实现为 C++级别。

[本模块的源代码](https://hg.python.org/cpython/file/3.5/Lib/this.py)
[我网站上的文章](https://adamoudad.github.io/post/zen-of-python/)