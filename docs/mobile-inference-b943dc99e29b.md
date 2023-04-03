# 量化和转换模型到 Tensorflow Lite 的指南

> 原文：<https://medium.com/analytics-vidhya/mobile-inference-b943dc99e29b?source=collection_archive---------2----------------------->

![](img/2482c1ca89341323bb8e2060691238cd.png)

近年来，在深度学习中，随着模型性能的提高，模型中的参数数量也在大量增加。例如— Inceptionv3 包含大约 23M 个参数。所有这些都使得在移动设备上运行它们变得不可能。

幸运的是，这样做有几个选择——设计轻量级架构，或者将训练模型的权重保存在低精度类型中，如 float16、uint8…

此外，我们将在 Tensorflow 的帮助下专注于量化和转换模型的第二个选项。

# 概观

在 Tensorflow Lite 中，有几个选项可用于获取移动优化模型

1.  从 Tensorflow 转换到 Tensorflow Lite，而不对权重和激活类型进行任何修改。
2.  转换模型并同时量化权重(使用量化算法中的默认范围)
3.  做量化感知训练，这将学习每层的最小和最大范围的重量，只有在转换成 tflite。

# 简单转换

在这种情况下，您的模型将在没有任何优化的情况下转换为 tflite，您将能够在移动设备上使用它，在 Tensorflow Lite 框架内进行推理。

下面是一个从 Keras 模型转换的例子。

```
converter = tf.lite.TFLiteConverter.from_keras_model_file('model_keras.h5')
tflite_model = converter.convert()
tflite_model_quant_file = **"./test.tflite"** tflite_model_quant_file.write_bytes(tflite_model)
```

# 转换+岗位培训量化

在这种情况下，您可以选择将权重转换为 uint8，这将减少模型的大小，但请注意，这可能会增加模型的延迟，因为在推理时间内，在对输入执行操作之前，层的权重将被转换回 float32，总之，此操作可能会花费大量时间，具体取决于您的架构。从张量流的官方文档来看，量子化的公式是

```
out[i] = (in[i] - min_range) * range(T) / (max_range - min_range)
if T == qint8: out[i] -= (range(T) + 1) / 2.0
where 
range(T) = numeric_limits<T>::max() - numeric_limits<T>::min()
```

此外，性能可能会下降，因为所有图层的 max_range 和 min_range 都有默认值。

这是一个使用训练后量化转换模型的示例。

```
converter = tf.lite.TFLiteConverter.from_keras_model_file('model_keras.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_model_quant_file = **"./test.tflite"** tflite_model_quant_file.write_bytes(tflite_model)
```

# 量化感知训练

为了解决下面提出的关于最小/最大值的默认范围的问题，Teansorflow 创建了一个流，在构建图形的过程中，您可以在每一层中插入假节点，以模拟向前和向后传递中的量化效果，并在训练过程中分别学习每一层的范围。

在训练中启用此选项所需的全部工作是在构建主模型后，添加转换图形的内置方法，以包括假节点

```
**with** tf.variable_scope(**'quantize'**):
    output= model(x=image_tf, is_training=**True**, keep_prob=keep_prob_tf)
tf.contrib.quantize.create_training_graph(quant_delay=0)
```

训练后，在导出图形的阶段，您需要调用一个方法将图形转换为相应的仅推理版本，其中已经训练的假节点将被视为每个层的最小/最大范围。

```
**with** tf.variable_scope(**'quantize'**):
    output = model(x=input, is_training=**False**, keep_prob=1.)g = tf.get_default_graph()
tf.contrib.quantize.create_eval_graph(input_graph=g)
```

之后，您可以在 TensorflowLite 转换器的帮助下转换模型，将推理选项和推理输入类型指定为 uint8。

```
converter = lite.TFLiteConverter.from_session(session, [input], [output])
converter.inference_type = tf.uint8
converter.inference_input_type = tf.uint8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}flatbuffer = converter.convert()**with** open(**'test.tflite'**, **'wb'**) **as** outfile:
    outfile.write(flatbuffer)
```

# 量化感知训练中的陷阱(针对 Tensorflow 1.14)

1.  不支持融合批处理规范，这是 tf.layers.batch_normalization 的默认选项。
2.  目前，train 和 eval 图中的变量范围存在差异，因此需要手动将变量范围添加到图中。

```
# the training stage
**with** tf.variable_scope(**'quantize'**):
    output= model(x=image_tf, is_training=**True**, keep_prob=keep_prob_tf)
tf.contrib.quantize.create_training_graph(quant_delay=0)# exporting stage
**with** tf.variable_scope(**'quantize'**):
    output = model(x=input, is_training=**False**, keep_prob=1.)
g = tf.get_default_graph()
tf.contrib.quantize.create_eval_graph(input_graph=g)
```

3.因为 TensorflowLite 转换器仅严格考虑输入和输出节点之间的节点，所以图中最后一层的伪节点(仅在实际节点之后)被忽略，这会导致 tflite 转换器出错。可以有一种变通方法，即添加一些无用的 op，这不会损害图形的结果，例如

```
**with** tf.variable_scope(**'quantize'**):
    output = model(x=input, is_training=**False**, keep_prob=1.)
    output = tf.maximum(output, -1e27)
```

[示例的源代码](https://github.com/lusinlu/tensorflow_lite_guide)

# 参考

[1]tensor flow Lite 的官方文档，【https://www.tensorflow.org/lite 

[2]张量流量化的实现，[https://www . tensor flow . org/API _ docs/python/TF/quantization/quantize](https://www.tensorflow.org/api_docs/python/tf/quantization/quantize)