# 伯努利和二项式分布

> 原文：<https://medium.com/analytics-vidhya/the-bernoulli-and-binomial-distributions-5d6247d48c6a?source=collection_archive---------5----------------------->

![](img/074dd75193ae54fa657a038adee8f13f.png)

一枚硬币的图像

# 概率分布—二项式分布

离散随机变量的概率可以用离散概率分布来概括。

> 离散概率分布用于机器学习，最显著的是用于二进制和多类分类问题的建模，但也用于评估二进制分类模型的性能，例如置信区间的计算，以及用于自然语言处理的文本中单词分布的建模。
> 
> *在为分类任务选择深度学习神经网络输出层的激活函数和选择合适的损失函数时，也需要离散概率分布的知识。*
> 
> *离散概率分布在应用机器学习中扮演着重要的角色，有一些分布是从业者必须了解的。*
> 
> *在本教程中，你会发现机器学习中使用的离散概率分布* ***(伯努利和二项式分布)*** *。*
> 
> *我们开始吧。*

# 随机变量

随机变量是由随机过程产生的量。

离散随机变量是一种随机变量，它可以有一组有限的特定结果。机器学习中最常用的两种离散随机变量是二进制和分类变量。

二元随机变量:x 在{0，1}分类随机变量:x 在{1，2，…，K}中。二元随机变量是离散随机变量，其中有限的结果集在{0，1}中。分类随机变量是离散随机变量，其中有限的结果集在{1，2，…，K}中，其中 K 是唯一结果的总数。

离散随机变量的每个结果或事件都有一个概率。

离散随机变量的事件与其概率之间的关系被称为离散概率分布，由**概率质量函数**或简称为 PMF 来概括。

对于可排序的结果，事件等于或小于给定值的概率由*累积分布函数*或简称 CDF 定义。

CDF 的逆函数称为百分点函数，将给出小于或等于某个概率的离散结果。

```
PMF: Probability Mass Function, returns the probability of a given outcome.
CDF: Cumulative Distribution Function, returns the probability of a value less than or equal to a given outcome.
PPF: Percent-Point Function, returns a discrete value that is less than or equal to the given probability.
```

常见的离散概率分布有很多。

最常见的是分别针对二元和分类离散随机变量的伯努利分布和多伯努利分布，以及将每个分布推广到多个独立试验的二项式分布和多项式分布。

在[3]中:

```
# for inline plots in jupyter
%matplotlib inline
# import matplotlib
import matplotlib.pyplot as plt
# for latex equations
from IPython.display import Math, Latex
# for displaying images
from IPython.core.display import Image
```

在[4]中:

```
# import seaborn
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
```

# 二项分布

伯努利分布是一种离散的概率分布，它涵盖了事件的二进制结果为 0 或 1 的情况。

{0，1}中的 x

“伯努利试验”是一个结果遵循伯努利分布的实验或案例。这个分布和试验是以瑞士数学家雅各布·伯努利的名字命名的。

伯努利试验的一些常见例子包括:掷一次硬币，结果可能是正面(0)或反面(1)。一胎生男孩(0)或女孩(1)。机器学习中伯努利试验的一个常见示例可能是将单个示例二进制分类为第一类(0)或第二类(1)。

伯努利分布只有两种可能的结果，即 1(成功)和 0(失败)，以及一次尝试，例如抛硬币。因此，具有伯努利分布的随机变量 X 可以取值 1 和成功概率 p，取值 0 和失败概率 q 或 1p，成功和失败的概率不一定相等。伯努利分布是二项分布的一种特殊情况，其中进行了一次试验(n=1)。

该分布可以用定义结果 1 的概率的单个变量 p 来概括。给定该参数，每个事件的概率可以计算如下:

```
P(x=1) = p
P(x=0) = 1 – p
In the case of flipping a fair coin, the value of p would be 0.5, giving a 50% probability of each outcome.
```

其概率质量函数由下式给出:

您可以使用 scipy.stats 模块的 bernoulli.rvs()方法生成伯努利分布的离散随机变量，该方法将 p(成功概率)作为形状参数。要改变分布，请使用 loc 参数。大小决定了重复试验的次数。如果希望保持再现性，请包含一个分配给某个数字的 random_state 参数。

在[5]中:

```
from scipy.stats import bernoulli
data_bern = bernoulli.rvs(size=10000,p=0.6)
```

再次可视化分布，您可以观察到您只有两种可能的结果:

在[6]中:

```
ax= sns.distplot(data_bern,
                 kde=False,
                 color="skyblue",
                 hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Bernoulli Distribution', ylabel='Frequency')C:\Users\user\New folder\lib\site-packages\scipy\stats\stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
  return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
```

Out[6]:

```
[Text(0, 0.5, 'Frequency'), Text(0.5, 0, 'Bernoulli Distribution')]
```

# 二项分布

多次独立伯努利试验的重复称为伯努利过程。

伯努利过程的结果将遵循二项式分布。因此，伯努利分布将是具有单一试验的二项分布。

机器学习算法在二元分类问题上的性能可以被分析为伯努利过程，其中模型对来自测试集的示例的预测是伯努利试验(正确或不正确)。

二项式分布总结了给定数量的伯努利试验 n 中的成功数量 k，以及每个试验 p 的给定成功概率。

只有两种可能结果的分布，如成功或失败，获得或损失，赢或输，并且所有试验的成功和失败概率都相同，这种分布称为**二项式分布**。然而，结果不一定是一样的，每个试验都是相互独立的。

二项分布的参数是 n 和 p，其中 n 是试验的总数，p 是每次试验成功的概率。其概率分布函数由下式给出:

其中:

# 二项分布方法综述

```
rvs(n, p, loc=0, size=1, random_state=None)---> Random variates.pmf(k, n, p, loc=0)---> Probability mass function.logpmf(k, n, p, loc=0)---> Log of the probability mass function.cdf(k, n, p, loc=0) ---> Cumulative distribution function.logcdf(k, n, p, loc=0) ---> Log of the cumulative distribution function.sf(k, n, p, loc=0) ---> Survival function (also defined as 1 - cdf, but sf is sometimes more accurate).logsf(k, n, p, loc=0) ---> Log of the survival function.ppf(q, n, p, loc=0) ---> Percent point function (inverse of cdf — percentiles).isf(q, n, p, loc=0) ---> Inverse survival function (inverse of sf).stats(n, p, loc=0, moments=’mv’) ---> Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’).entropy(n, p, loc=0) ---> (Differential) entropy of the RV.expect(func, args=(n, p), loc=0, lb=None, ub=None, conditional=False) ---> Expected value of a function (of one argument) with respect to the distribution.median(n, p, loc=0) ---> Median of the distribution.mean(n, p, loc=0) ---> Mean of the distribution.var(n, p, loc=0) ---> Variance of the distribution.std(n, p, loc=0) ---> Standard deviation of the distribution.interval(alpha, n, p, loc=0) ---> Endpoints of the range that contains alpha percent of the distribution
```

# 为二项分布生成随机变量

您可以使用 scipy.stats 模块的 binom.rvs()方法生成二项式分布离散随机变量，该方法将 n(试验次数)和 p(成功概率)作为形状参数。要改变分布，请使用 loc 参数。大小决定了重复试验的次数。如果希望保持再现性，请包含一个分配给某个数字的 random_state 参数。

在[7]中:

```
from scipy.stats import binom
data_binom = binom.rvs(n=10,p=0.5,size=10000)
```

在[8]中:

```
ax = sns.distplot(data_binom,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
```

Out[8]:

```
[Text(0, 0.5, 'Frequency'), Text(0.5, 0, 'Binomial Distribution')]
```

或者，可以调用分布对象(作为一个函数)来确定形状和位置。这将返回一个“冻结的”RV 对象，其中包含固定的给定参数。

冻结分配并显示冻结的 pmf:

在[30]中:

```
import numpy as np
fig, ax = plt.subplots(1, 1)
n=10
p=0.5
x = np.arange(0,11)# ppf is the inverse of cdf
ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')
ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)#plot the frozen binomial distribution
rv = binom(n,p)
ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
        label='frozen pmf')ax.legend(loc='best', frameon=False)
plt.show()
```

在[31]中:

```
x
```

Out[31]:

```
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
```

我们可以使用 binom.stats() SciPy 函数计算这个分布的矩，特别是期望值或均值和方差。

*μ* = *np*

根据上述二项式分布(也等于分布的平均值)，预计 10 次试验中有 5 次成功的概率最高。

在[32]中:

```
n, p = 10, 0.5
mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')
print('Mean=%.3f,Variance=%.3f'%(mean,var) )Mean=5.000,Variance=2.500
```

检查 cdf 和 ppf 的准确性:

在[45]中:

```
prob = binom.cdf(x, n, p)
np.allclose(x, binom.ppf(prob, n, p))
```

Out[45]:

```
True
```

让我们使用 binom.cdf 函数手动检查

在[46]中:

```
for k in range(11):
    print('P of %d successes in 10 trials: %.3f%%' %(k, binom.pmf(k,n,p)*100))P of 0 successes in 10 trials: 0.098%
P of 1 successes in 10 trials: 0.977%
P of 2 successes in 10 trials: 4.395%
P of 3 successes in 10 trials: 11.719%
P of 4 successes in 10 trials: 20.508%
P of 5 successes in 10 trials: 24.609%
P of 6 successes in 10 trials: 20.508%
P of 7 successes in 10 trials: 11.719%
P of 8 successes in 10 trials: 4.395%
P of 9 successes in 10 trials: 0.977%
P of 10 successes in 10 trials: 0.098%
```

这可以比作抛硬币试验，其中 n= 10，成功被定义为获得正面。上面我们问的问题是“在 k 为 0，1，2…10 的情况下，n 次试验得到 k 个头的概率是多少”。请注意，*P*(*k*= 0)=*P*(*k*= 10)P(k = 0)= P(k = 10)对于 p =0.5 的公平硬币

在[47]中:

```
for k in range(11):
    print('P of %d success in 10 trials: %.3f%%' %(k, binom.cdf(k,n,p)*100))P of 0 success in 10 trials: 0.098%
P of 1 success in 10 trials: 1.074%
P of 2 success in 10 trials: 5.469%
P of 3 success in 10 trials: 17.187%
P of 4 success in 10 trials: 37.695%
P of 5 success in 10 trials: 62.305%
P of 6 success in 10 trials: 82.812%
P of 7 success in 10 trials: 94.531%
P of 8 success in 10 trials: 98.926%
P of 9 success in 10 trials: 99.902%
P of 10 success in 10 trials: 100.000%
```

运行上面的代码定义了二项式分布，并使用 binom.pmf 函数计算每个成功结果的概率。记住 x 是一个从 1 到 10 的数组，包括 1 和 10。

我们可以看到，5 个成功结果的概率最高，约为 24.609%

假设一次试验的成功概率为 50%，我们预计 10 次试验中有 10 次或更少成功的概率接近 100%。我们可以用累积分布函数来计算，如上所示。

# 重复抛硬币实验

在上面的例子中，我们投掷一枚硬币“n=10”次。这是我们唯一的实验。通常，为了看看我们的掷硬币实验有多可靠，我们可能想要多次重复这个实验(或者考虑投掷多个硬币)。我们可以通过 numpy.random.binomial 函数或 scipy.binom.rvs 函数中的“大小”选项轻松模拟多个实验

让我们重复掷硬币实验 100 次，其中每个实验我们掷 10 次公平的硬币。让我们问一下，在 100 次实验中，我们每次看到多少个头。

在[86]:

```
np.random.binomial?
```

在[88]中:

```
x= np.random.binomial(n,p,size=100)
n=10
p=0.5
# let us repeat our experiment for 100 times
size = 100
x
```

Out[88]:

```
array([5, 5, 7, 3, 5, 4, 5, 5, 7, 7, 6, 6, 6, 5, 2, 2, 7, 4, 4, 5, 2, 8,
       3, 8, 6, 7, 2, 5, 8, 6, 5, 6, 4, 6, 4, 4, 5, 2, 2, 2, 6, 2, 3, 5,
       5, 4, 4, 7, 3, 7, 4, 5, 8, 6, 4, 8, 6, 4, 5, 5, 5, 3, 6, 4, 4, 5,
       4, 7, 5, 3, 3, 6, 7, 3, 5, 6, 7, 2, 4, 3, 5, 4, 5, 6, 8, 3, 6, 7,
       5, 6, 4, 2, 3, 3, 7, 8, 5, 6, 4, 5])
```

在[89]:

```
ax = sns.distplot(x,
                  kde=False,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
```

Out[89]:

```
[Text(0, 0.5, 'Frequency'), Text(0.5, 0, 'Binomial Distribution')]
```

在 n=10 次抛硬币中看到 x 个正面的概率我们从一个简单的实验开始，将一枚公平的硬币抛 10 次。我们将实验重复了 100 次，并测量了我们观察到的成功次数/人头数。我们可以用多种方式观察到的成功次数(人头数)来理解概率的基础。例如，我们可以简单地数一数在掷硬币的过程中，我们看到了多少次 0 头、1 头、2 头，等等。

我们可以看到，在我们的 100 次实验中，我们从来没有看到所有正面和所有反面(因为第一个和最后一个元素都是零)。我们还可以看到，我们观察到更多次 4，或 5，或 6 个头。上述成功的总数为 100，即我们的实验总数。我们可以用观察到的成功除以 100 来估计在 n=10 次抛硬币中获得 x 次成功的概率。

在[90]:

```
probs_100 =[sum(np.random.binomial(n,p,size=100) == i)/100 for i in range(n+1)]
probs_100
```

Out[90]:

```
[0.0, 0.0, 0.07, 0.1, 0.22, 0.25, 0.18, 0.12, 0.03, 0.01, 0.0]
```

让我们画出我们刚刚计算的 x 成功的概率。

在[91]:

```
plt.xticks(range(n+1))
plt.plot(list(range(n+1)), probs_100, color='blue', marker='o')
plt.xlabel('Number of Heads',fontsize=14)
plt.ylabel('Probability',fontsize=14)
```

Out[91]:

```
Text(0, 0.5, 'Probability')
```

从上面的剧情我们可以看出，看到 5 个头的概率是最高的。注意，这个观察结果可能会根据我们随机模拟的实现而变化。

# 10 万次重复实验

我们知道这是一个公平的硬币，所以我们预计，如果我们重复实验更多次，我们应该看到看到 5 个头的概率应该是最高的。所以，让我们重复我们的掷硬币实验 100，000 次，并计算看到 n 个像上面这样的头的概率。此外，请记住，观察结果可能会有所不同，因为这是一个随机过程。

在[118]:

```
from ipywidgets import interact,IntSlider
def plot_probab(size=100):
    np.random.seed(42)
    n=10
    p=0.5
# let us repeat our experiment for 100000 times x=np.random.binomial(n=n, p=p, size=size)
    probs_100= [sum(np.random.binomial(n,p,size=size) == i)/size for i in range(n+1)]
    plt.xticks(range(n+1))
    plt.plot(list(range(n+1)), probs_100, color='blue', marker='o')
    plt.xlabel('Number of Heads',fontsize=14)
    plt.ylabel('Probability',fontsize=14)

size_slider = IntSlider(min=100, max=100000, step=100, value=100, description='$\\nu$')   
interact(plot_probab,size=size_slider);
```

现在我们可以看到，看到 5 个头的概率是我们预期的最高。巧妙的是，即使我们不知道硬币是否公平，但如果我们像上面一样反复做实验，观察成功的次数，我们就可以推断出硬币是否公平。

# 扔有偏见的硬币

让我们用有偏差的硬币做实验。假设我们有一个硬币，我们怀疑它是一个有偏差的硬币。让我们像以前一样通过反复实验来推断硬币有多不公平。

就像之前描述的那样，让我们把不公平的硬币抛 10 次，重复 10 万次，数一数成功的次数。让我们用成功的次数来得到 x 次成功的概率，并把它画出来。

在[119]:

```
def plot_probab_unfair(size=100):
    np.random.seed(42)
    n=10
    p=0.7
# let us repeat our experiment for 100000 times x=np.random.binomial(n=n, p=p, size=size)
    probs_100= [sum(np.random.binomial(n,p,size=size) == i)/size for i in range(n+1)]
    plt.xticks(range(n+1))
    plt.plot(list(range(n+1)), probs_100, color='blue', marker='o')
    plt.xlabel('Number of Heads',fontsize=14)
    plt.ylabel('Probability',fontsize=14)

size_slider = IntSlider(min=100, max=100000, step=100, value=100, description='$\\nu$')   
interact(plot_probab_unfair,size=size_slider);
```

从上面的图中我们可以看出，当成功/人头数为 7 时，成功的概率最高。因此，我们可以推断，偏硬币的成功概率 p=0.7。

# 二项式分布的应用(现实生活中的例子)

一家公司钻了 9 口野猫石油勘探井，每口井的成功概率估计为 0.1。九口井全部失败。发生这种情况的可能性有多大？

让我们对模型进行 20，000 次试验，并计算产生零阳性结果的数量。

在[122]:

```
sum(np.random.binomial(9, 0.1, 20000) == 0)/20000.
# answer = 0.38885, or 38%.
```

Out[122]:

```
0.38495
```

参考

1.  杰森·布朗利的掌握数据科学博客
2.  Scipy。统计数据