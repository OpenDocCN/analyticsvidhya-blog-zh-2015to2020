# 模型优化技术

> 原文：<https://medium.com/analytics-vidhya/model-optimization-techniques-79a3a96b6427?source=collection_archive---------15----------------------->

![](img/0a89e2904d9efa86a6fe971eca3f20a3.png)

Tensorflow 2.3 已经发布了，请到他们的网站上看看

训练结束后，当我看着我的模特尺寸时，我总是因为它显示的数字而感到沮丧。但是没有更多的担忧，因为机器学习世界在过去十年里已经有了巨大的发展。在本文中，我将向您介绍使用 TensorFlow 进行模型优化的方法。

**Tensorflow** 是谷歌创建的开源深度学习框架，它应该在每个机器学习研究人员和有志者的列表中。可以说，在这十年里，每一个新模型都是基于 Tensorflow 或 Pytorch。最初，市场上还有其他参与者，如 **MXNet、Caffe** 等，但现在与 Tensorflow 和 Pytorch 相比，它们的份额非常少。随着权重聚类被添加到已经存在的剪枝和量化中，有三种主要的方法可以在不减少度量的情况下减少他们的模型的大小。

**修剪:**

修剪实质上是取消神经网络的一些权重。所以这里发生的是在训练过程中，一些神经元没有被训练完，只有重要的神经元被训练。当无效神经元不再参与推理时，情况也是如此。修剪有助于显著压缩模型，即它可以减少高达 6 倍的大小，并且指标损失 0.1%。以下是在以常规方式编译和训练模型后，如何在 Keras 模型中实现修剪的方法。

```
import tensorflow_model_optimization as tfmotprune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set. num_images = train_images.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}model_for_pruning = prune_low_magnitude(model, **pruning_params)# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])model_for_pruning.summary()
```

**量化:**

在这种方法中，我们不会取消任何神经元，但我们会将模型权重从浮点数转换为整数。实践中主要有两种量化方法。一个是量化感知训练，另一个是后量化。量化感知训练包括在训练期间将权重转换为整数，而后量化是在保存模型时将权重转换为整数。下面是在 Keras 模型中实现这两种方法的方法

*   **量化感知训练**

```
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

q_aware_model.summary()
```

*   **后量化**

```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

**权重聚类**

这是我更感兴趣的方法，因为它的实现方式。将单个层中的权重聚类成 ***N*** 个聚类，并找到它们各自的质心值。这些权重随后被质心值取代，这将极大地有助于模型压缩。与以前的方法相比，获得的度量更好。模型度量和大小取决于 N 值，如果它太低，我们将得到高度压缩的模型，但度量将非常差，反之亦然。下面是它在 Keras 模型中的实现方式

```
``import tensorflow_model_optimization as tfmot

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
  'number_of_clusters': 16,
  'cluster_centroids_init': CentroidInitialization.LINEAR
}

# Cluster a whole model
clustered_model = cluster_weights(model, **clustering_params)

# Use smaller learning rate for fine-tuning clustered model
opt = tf.keras.optimizers.Adam(learning_rate=1e-5)

clustered_model.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer=opt,
  metrics=['accuracy'])

clustered_model.summary()
```

所以这是我第一次尝试从我所涉足的领域解释一些东西。我希望你觉得这是有用的。谢谢你阅读它。您的评论比以往任何时候都受欢迎，我感谢您的时间和努力

请继续关注我的账户，我会经常发布新的文章