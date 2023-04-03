# 在 Matlab 上对信号进行重采样时的边缘效应。怎么解决？

> 原文：<https://medium.com/analytics-vidhya/edge-effects-when-resampling-a-signal-on-matlab-how-to-solve-it-cc6458ab1306?source=collection_archive---------6----------------------->

人们普遍认为真正的学习是建立在经验基础上的。当你在开发信号处理应用程序时，即使使用 Matlab 这样强大的软件工具，有时也会出现意想不到的效果，我们只是能够用实际经验来看。这就是为什么说得好，“魔鬼在细节中”。

在这篇文章中，我将介绍当信号需要重新采样时出现的不良效应，以及如何用 Matlab 来解决这一问题。此外…