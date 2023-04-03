# 第 1 部分:用 Python 模拟随机行走

> 原文：<https://medium.com/analytics-vidhya/simulating-random-walk-in-python-ed95bfb55210?source=collection_archive---------6----------------------->

## 在本文中，我将简要讨论随机森林，并用 Python 编写代码来模拟这个概念。

稍后，我们将通过**模拟细菌试图使用随机漫步寻找食物来扩展学习。**

![](img/1b6d22fcef5ca8221ed697c2ede409e6.png)

首先关于随机行走，它基本上是一个物体从起点随机行走的过程。

这个概念可能看起来微不足道，但我们可以将自然界中的许多现象和行为与“随机行走”联系起来

这个概念在各个领域都有应用。

[](https://en.wikipedia.org/wiki/Random_walk#Applications) [## 随机漫步应用

en.wikipedia.org](https://en.wikipedia.org/wiki/Random_walk#Applications) 

我想在 Python 上模拟这个概念，并可视化地绘制和查看代码运行。

你可以在我的 [GitHub 页面](https://gist.github.com/saikumar-solowarrior/96efb15ec34ee563d7d8e9706f31a97f)找到完整的代码。

## 解释代码:

## 第 1 部分:导入所需的模块

```
*import* numpy *as* np
*import* matplotlib.pyplot *as* plt
*import* matplotlib.animation *as* animation
```

这里我们使用了两个模块，Numpy 和 Matplotlib。

## 第 2 部分:设置和模拟随机漫步

```
*#setting up steps for simulating 2D*dims = 2
step_n = 200
step_set = [-1, 0, 1]
origin = np.zeros((1,dims))*#Simulate steps in 2D* step_shape = (step_n,dims)
steps = np.random.choice(*a*=step_set, *size*=step_shape)
path = np.concatenate([origin, steps]).cumsum(0)
```

这里解释一下代码，变量“ **dims** ”指的是维度，随机游走可以在 1 维、2 维和 3 维中模拟。在这里，我在二维空间中模拟它，因此“dims = 2”。

**步数::**这是指我们要分配给随机行走的步数。我想让它随机走 200 步。

**origin ::** 使用 Numpy.zeros 获得大小为 1*2 (1 行 2 列)的零数组。

**step_shape::** 原点和 step_set 的元组。这实际上存储了我们需要的结果数组的大小，以适应我们将要模拟的数据。

例如，如果我们要执行 50 个步骤，我们将指定 step_n=50，step_shape = (50，2)，这是我们以后需要的数组大小。

**步骤::**我们使用 Numpy 函数 random.choice 从给定的 1D 数组中生成数组的随机样本，样本大小为 step_change。

因此，由于给定的输入是步长集(包含-1，0 和 1)，输出将是一个包含-1，0 和 1 的数组，大小为**步长集**。

**path::我一次完成了两个步骤，沿着 0 轴连接并执行累积求和。**

查看文档以详细了解这些方法。

***现在，我们有了随机漫步的模拟结果。***

> 如果你需要关于这一步的信息或解释，你可以在评论中写下来，这样我就可以试着贴出来。

## 第三步:绘图和可视化。

现在，为了可视化我们刚刚模拟的随机行走数据，我们将使用“matplotlib”库。

```
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, c=”green”)
ax.set_ylim(-10, 10)
ax.set_xlim(-5, 10)
plt.title(‘2D Random Walk’)xdata, ydata = [], []
del xdata[:]
del ydata[:]
line.set_data(xdata, ydata)
```

上面的代码应该允许我们在图上绘制模拟的随机行走。

然而，我的目标不是放一个静态的图，而是输出我的随机漫步代码的现场模拟。

**第四步:制作随机漫步的动画**

关于如何做到这一点，我参考了 Matplotlib 的文档和示例。Matplotlib 通过提供动画类提供了实现这一点的方法。

matplotlib。动画用于获得现场模拟。

我可以写关于 matplotlib 的详细描述。动画，以后我会单独写一篇关于这个的文章供参考。

结果是这样的:

步数为 300 时的随机漫步

页（page 的缩写）s:我添加了代码，如果随机漫步超出了框架，就重新绘制图形。

> 第 2 部分:细菌及其与有偏随机游动的关系

在第 2 部分中，我将使用随机漫步创建一个细菌模拟，细菌将尝试使用该技术找到食物来源。

随机漫步有多种应用，但这个作为一个快速有趣的项目让我着迷。