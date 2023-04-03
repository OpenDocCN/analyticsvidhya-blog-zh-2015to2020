# 使用 TensorFlow 可视化激活热图

> 原文：<https://medium.com/analytics-vidhya/visualizing-activation-heatmaps-using-tensorflow-5bdba018f759?source=collection_archive---------0----------------------->

当卷积神经网络进行预测时，可视化卷积神经网络的值可能是有益的，因为它允许我们看到我们的模型是否在轨道上，以及它发现哪些特征是重要的。例如，在确定图像是否是人类时，我们的模型可能会发现面部特征是决定性因素。

为了可视化热图，我们将使用一种称为 Grad-CAM(梯度类激活图)的技术。背后的想法相当简单；为了找到某个类在我们的模型中的重要性，我们简单地取它的梯度…