# 为什么最速下降的方向总是与损失函数的梯度相反？

> 原文：<https://medium.com/analytics-vidhya/why-direction-of-steepest-descent-is-always-opposite-to-the-gradient-of-loss-function-dddc995a816e?source=collection_archive---------19----------------------->

我们都听说过梯度下降算法，以及它是如何在每次迭代中以最小化损失函数的方式更新参数的。并且，最陡的下降是当损失函数最小时。但是你知道为什么最陡下降总是和损失函数的梯度相反吗？或者为什么我们称算法为‘梯度下降’？让我们来了解一下！

**w(t) = w(t-1) -α(∇L)**

在这个更新等式中，- **∇L** 是“相反”方向。但是为什么呢？

## 都在泰勒级数里！

让我们考虑一下 **w** 的一个小变化。更新后的权重将变成， **w(新)= w+ξδw**。(记住 w 是矢量，所以**δw**是方向的变化， **η** 是变化的幅度)。

现在，让**δw = u**和**转置(u) = v** 。新的损失函数 w.r.t **w(new)** 将是 **L( w + ηu)** 。

假设是 **w** 的一个小变化，我们可以通过泰勒展开写出新的损失函数如下:

**L(w + ηu) = L(w) + (η)。v∇L(w) + (η /2！).v∇大学……**

现在，η非常小。所以，我们可以忽略包含η的项，η和后面的项。所以，我们的新方程变成了，

**L(w + ηu)-L(w) = (η)。v∇L(w)**

正如我们所知，我们的目标是在每一步中最小化损失函数。因此，

L(w + ηu)-L(w) < 0 即新损失小于旧损失。从上面的等式我们可以说， **v.∇L(w) < 0**

**v.∇L(w)** 是点积。设 **β** 为 **v** 与**∇l(w**之间的角度

cos(**β)=(v.∇l(w))/|v||∇l(w)|.**为简单起见，让 **|v||∇L(w)|=k** 。

因此，cos( **β) = (v.∇L(w))/k，**其中 **-1≤** cos( **β)≤1。**

现在，我们希望**【v.∇l(w】**尽可能低/负(我们希望我们的新损失尽可能小于旧损失)。因此，我们希望 cos( **β)** 尽可能低。cos( **β)** 可以取的最小值是-1。这种情况下， **β = 180 度。**

因此，为了最大程度地最小化损失函数，算法总是转向与损失函数的梯度相反的方向， **∇L**

![](img/f51b407cf220ce54942e5dc3f39e7c5c.png)

[https://ml-cheat sheet . readthedocs . io/en/latest/_ images/gradient _ descent _ de mysticed](https://ml-cheatsheet.readthedocs.io/en/latest/_images/gradient_descent_demystified.png)

*注:本文的概念基于 NPTEL 在线教授的课程* [*CS7015:深度学习*](https://www.cse.iitm.ac.in/~miteshk/CS7015.html) *的视频。*

## 资源:

1.  [泰勒级数](https://en.wikipedia.org/wiki/Taylor_series#:~:text=In%20mathematics%2C%20the%20Taylor%20series,are%20equal%20near%20this%20point.)
2.  梯度