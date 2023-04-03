# 用神经网络求解微分方程

> 原文：<https://medium.com/analytics-vidhya/solving-differential-equation-by-a-neural-network-6a67d2a29d16?source=collection_archive---------9----------------------->

我们可以在商业、零售、计算机视觉和自然语言处理中找到神经网络的大量应用，但是对于那些不是数据驱动而是严重依赖于微分方程解的工程领域呢？由于一篇突破性的论文——“物理学通知深度学习(第一部分):非线性偏微分方程的数据驱动解决方案”，现在我们可以使用神经网络来解决微分方程。作者们也在他们的 Github 资源库中慷慨地分享代码，但这些代码是在 Tensorflow 的以前版本中编写的。

让我们开始理解和实现他们的概念。对于这个博客，我选择了一个简单的微分方程。

![](img/080e0cfd42136448c14738d09ede96cd.png)

轴向载荷下杆的轴向位移方程

对于那些不熟悉的人来说，这个方程是杆在轴向载荷下的轴向位移。a 是横截面积，E 是杨氏模量，q 是力。对于那些不熟悉这些词的人来说，不必担心。这些可以被认为是等式的常数。如图所示，这个微分方程有两个边界条件。

本文采用一种独特的概念，将微分方程和边界条件相结合，形成一个定制的代价函数，并进一步优化该代价函数。在写最后一个术语之前，让我们定义一些术语。

![](img/0b0d34f8aac3257098396402141a838a.png)

成本函数形成

这个概念很好，但是如何在软件中实现呢？答案是用张量流的自动微分法。让我们看一些代码片段来理解

```
def f_model(self):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.x_f)#collocation points 
      tape.watch(self.x_u_l)#boundary condition on u
      tape.watch(self.x_u_b)#boundary condition on ux
      # Packing together the inputs
      X_f = self.x_f
      X_b=self.x_b
      X_u=self.x_u
      # Getting the prediction
      u = self.u_model(X_f)
      u_b=self.u_model(X_b)
      u_u=self.u_model(X_u)
      # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
      u_x = tape.gradient(u, self.x_f)
      u_u_x = tape.gradient(u_u, self.x_u)
    # Getting the other derivatives
    u_xx = tape.gradient(u_x, self.x_f)
    # Letting the tape go
    del tape
    # Buidling the PINNs
    return tf.reduce_mean(tf.square(self.A*self.E*u_xx+self.q))+8*tf.reduce_mean(tf.square(u_u_x+u_b))
```

Tape 是 Tensorflow 2.0 中的更新，它可以帮助我们计算梯度。在代码片段中，我们首先从模型中计算 u(位移),然后使用 tape 计算 Ux(位移相对于 x 的导数)。通过使用这个 Uxx(双导数)被计算。我们必须记住，边界条件也有 U 和 Ux 项，所以这些项需要为边界点再次计算。

写完这个函数后，我们只需建立这个目标函数或成本函数，并将其最小化。与其他神经网络问题不同，我们必须**过拟合**这个问题。创建一个深度神经网络并运行它足够长的时间以获得目标函数的最小值。由于使用了二重导数，我们不能使用 RELU，因为 RELU 的二阶导数将为零。参考文献使用了 [**Tanh**](https://github.com/pierremtb/PINNs-TF2.0) 作为激活函数。

代码执行参见-[https://github.com/pierremtb/PINNs-TF2.0](https://github.com/pierremtb/PINNs-TF2.0)。

![](img/b3d5f71b36a74709cfe8b1a56050331d.png)

神经网络的最终结果