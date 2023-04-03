# 逐步张量流/ Keras

> 原文：<https://medium.com/analytics-vidhya/step-by-step-tensorflow-keras-295ed1647620?source=collection_archive---------41----------------------->

第 2 部分:CNN /数学背景

在用 Keras 实现卷积神经网络模型之前，我的目的是从卷积的数学背景开始。

![](img/494e9c41a2ef24cb732463f9f8b50740.png)

**算法**
*1)使用位置—内核窗口的反转…
2)根据内核大小分割数据…
3)计算数据块和反转内核窗口之间的点积…
4)计算输出的大小…*

![](img/aa90ec67d1ac63fa7eafce614664456b.png)

```
***Size of Convolution:***Size of X : 𝑆𝑥
Size of W : 𝑆𝑤
Size of Output : 𝑆𝑥–𝑆𝑤+1𝑆𝑥–𝑆𝑤 +1 => 5–3+1 = 3
```

**卷积的功能**
*1)填充:用零增加元素…
2)跨越:内核窗口移位…*

![](img/088d4cbe12ae9f1a2886333670102bc1.png)

```
***Size of Padding:***Size of X : 𝑆𝑥
Size of W : 𝑆𝑤
Padding = P
Size of Ouput : 𝑆𝑥+2P–𝑆𝑤+1P = 1 => 𝑆𝑥 + 2P – 𝑆𝑤 + 1 = 5 + 2 – 3 + 1 = 5
```

![](img/55b127b7ec33782c6e055a9899593820.png)

```
***Size of Striding:***Size of X : 𝑆𝑥
Size of W : 𝑆𝑤
Stride : S
Size of Output : ((𝑆𝑥−𝑆𝑤)/𝑆)+1𝑆𝑥 = 5, 𝑆𝑤 = 3, S = 2 => Size of Output = 2
```

![](img/8956976134283d484bb2124859e86dfd.png)

```
***Size of Padding & Striding:***Size of X : 𝑆𝑥
Size of W : 𝑆𝑤
Padding : P
Stride : S
Size of Output : ((𝑆𝑥+2𝑃−𝑆𝑤)/𝑆)+1𝑆𝑥 = 6, 𝑆𝑤 = 3, P = 1, S = 2 => Size of Output = 3
```

![](img/17be24aceeec8a926915865d11b32701.png)

```
X = [1, 2, 3, 4, 5, 6]
Pooling Size = 3***On Algorithm :***1) Pooling Size = 3 => Striding = 3
2) Chunks of X => [1, 2, 3], [4, 5, 6]
3) [1, 2, 3] => 3; [4, 5, 6] => 6
4) Pooling Output = [3, 6]***On Algorithm :***1) Pooling Size = 3 => Striding = 2
2) Chunks of X => [1, 2, 3], [3, 4, 5]
3) [1, 2, 3] => 3; [4, 5, 6] => 5
4) Pooling Output = [3, 5]***Size of :***Size of X : 𝑆𝑥
Size of W : 𝑆𝑤
Pooling : P
Stride : S (S ≤ P)
Size of Output : 𝑆𝑥 / P
𝑆𝑥 = 6, 𝑆𝑤 = 3, Pool = 3, S = 3 => Size of Output = 2
𝑆𝑥 = 6, 𝑆𝑤 = 3, Pool = 3, S = 2 => Size of Output = 2
```

![](img/37ee7b7b416222b961dc3668af1160a1.png)

```
***Variants of Pooling***1) Chunks and Maximum
2) Padding, Chunks and Average
3) Padding, Chunks and Average without padding elementsSamples of “Chunks and Maximum” are above…
```

![](img/2b51785bdfcd59d43a9b4cc86c7200b9.png)![](img/f7e79579d2cfaf5836e246fe5eafea19.png)![](img/7af403e3f32271539de32bdd6bb84459.png)![](img/ce2ae3a83fe130ff5688970a5377404c.png)

**通道**
这个关键字是这个问题的答案:“你的数据(一般是图像数据)有多少个通道？”
如果您的数据具有体积维度(通道维度，如 RGB)，您的内核窗口应该具有第三个(通道)维度…
此外，如果我们将使用两个或更多过滤器，那么，我们必须向输出添加一个维度…
过滤器的数量和通道的数量是完全不同的参数…

![](img/ddfa5369b2f963d8dfa4cadcc4929a12.png)![](img/33c653059499fa19a08659aac2b89d85.png)

```
***Formula***Input : X
Kernel : W
Bias : b
Activation Function : f
Output : 𝑃𝑜𝑜𝑙(𝑓(𝑋∗𝑊+𝑏))
```

![](img/1821e842bf5b4a176cb7ba40bff243ff.png)

# CNN 的应用

图像分类
图像识别
物体检测
视频分类
NLP
时域异常检测
药物发现

# 然后

我会在 TensorFlow / Keras 上设计一个卷积神经网络。

# 参考

> *deeplearning.ai 专业化讲义，Coursera*
> 
> *tensor flow in practice specialty 讲义，Coursera*
> 
> *IBM AI 工程专业化讲义，Coursera*
> 
> *Deniz Yuret 深度学习介绍讲义，Koc 大学*