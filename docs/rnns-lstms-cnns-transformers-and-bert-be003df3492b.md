# RNNs、LSTMs、CNN、Transformers 和 BERT

> 原文：<https://medium.com/analytics-vidhya/rnns-lstms-cnns-transformers-and-bert-be003df3492b?source=collection_archive---------1----------------------->

# 递归神经网络

rnn 确实有记忆来跟踪事物，所以它们允许信息在网络上持久存在。看下面给出的图片。图像的左侧显示了一个 RNN 单元，它接受一些输入，比如 x，并在隐藏单元 h 中进行处理，最终以输出 o 做出响应。除了线性之外，隐藏层中的循环还允许将数据发送到下一层。总的来说，我们可以说 RNN 是一组相似的网络块。如果我们展开左边的图像，我们会得到…