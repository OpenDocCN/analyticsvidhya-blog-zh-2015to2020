# C++中推理 Tensorflow2 模型

> 原文：<https://medium.com/analytics-vidhya/inference-tensorflow2-model-in-c-aa73a6af41cf?source=collection_archive---------4----------------------->

这篇文章是关于 Tensorflow2 模型的 C++推理。注意，这是关于 **Tensorflow Lite** 的**而不是**(你可以在官方页面找到一个不错的指南:【https://www.tensorflow.org/lite/guide】)。

# 快速启动

1.  以 SavedModel 格式([https://www.tensorflow.org/guide/saved_model](https://www.tensorflow.org/guide/saved_model))导出训练好的模型
2.  使用 SavedModel CLI 查找输入和输出张量名称([https://www . tensor flow . org/guide/saved _ model # details _ of _ the _ saved model _ command _ line _ interface](https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface))
3.  使用 Tensorflow C++ API 的设置环境([https://github.com/FloopCZ/tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc))
4.  负载和推理模型

# 以 SavedModel 格式导出训练模型

对于 C++推理，您应该做的第一件事是导出模型。Tensorflow 提供了一个名为 **SavedModel** 的格式，包含了运行模型的所有需求。以 **SavedModel** 格式导出模型的 python 代码如下。(更多详情:[https://www.tensorflow.org/guide/saved_model](https://www.tensorflow.org/guide/saved_model))

# 使用 SavedModel CLI 查找输入和输出张量名称

对于 C++推理，我们需要输入和输出张量的准确名称。我们可以使用 Tensorflow 的 SavedModel CLI 从导出的模型中解析这些信息。该命令如下所示:

这会打印出张量名称，如下所示:

从上面的结果，我可以找到输入和输出张量的名称。

> 输入:服务 _ 默认 _ 输入 _ 张量:0
> 
> 输出:stateflupartitionedcall:0，stateflupartitionedcall:1，stateflupartitionedcall:2，stateflupartitionedcall:3，stateflupartitionedcall:4，stateflupartitionedcall:5

# 使用 Tensorflow C++ API 的设置环境

Tensorflow C++ API 需要 Tensorflow 源代码文件夹和 Bazel 构建系统。然而，FloopCZ 共享了一个很好的存储库，使得 Tensorflow C++ API 没有这些需求成为可能。请跟随这里的向导:[https://github.com/FloopCZ/tensorflow_cc](https://github.com/FloopCZ/tensorflow_cc)

您可以使用 docker 容器进行快速设置([https://hub.docker.com/r/floopcz/tensorflow_cc/](https://hub.docker.com/r/floopcz/tensorflow_cc/))。

# 负载和推理模型

Tensorflow C++ API 为**saved model:**[https://github . com/tensor flow/tensor flow/blob/master/tensor flow/cc/saved _ model/loader . h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h)

函数 **LoadSavedModel** 返回 **SavedModelBundleLite** 类的一个实例，其中包含用于运行模型的**会话**。在官方[页面](https://www.tensorflow.org/guide/saved_model#load_a_savedmodel_in_c)中，示例代码使用 **SavedModelBundle** ，但更倾向于使用 **SavedModelBundleLite** (请阅读 loader.h 中 **SavedModelBundleLite** 类的注释)。

要运行模型，首先调用**SavedModelBundleLite::GetSession()**来获取会话实例。然后，像往常一样，推理模型通过调用 **Session::Run()** 。这里我们提供了从 **SavedModel CLI** 获得的输入和输出张量的名称。