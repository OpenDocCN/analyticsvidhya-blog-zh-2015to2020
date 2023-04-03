# 英特尔 OpenVINO:模型优化器

> 原文：<https://medium.com/analytics-vidhya/intel-openvino-model-optimizer-e381affa458c?source=collection_archive---------5----------------------->

在我的[上一篇文章](/swlh/introduction-to-intel-openvino-toolkit-5f98dbb30ffb)中，我已经讨论了 OpenVINO 工具包的基础和工作流程。在这篇文章中，我们将探索:-

*   什么是模型优化器？
*   配置模型优化器
*   将 ONNX 模型转换为中间表示
*   将 Caffe 模型转换为中间表示
*   将张量流模型转换为中间表示

![](img/48edc9a7968773f116c63b6d69d82c5b.png)

# 什么是模型优化器？

模型优化器是 OpenVINO 工具包的两个主要组件之一。模型优化器的主要目的是将模型转换为中间表示(IR)。模型的中间表示(IR)包含一个**。xml 文件**和一个**。bin** 文件。您需要这两个文件来运行推理。

*   **。xml** - >包含了模型架构的其他重要元数据。
*   **。bin** - >包含二进制格式的模型权重和偏差。

中间表示(IRs)是 OpenVINO Toolkit 的标准结构和神经网络架构的命名。TensorFlow 中的“Conv2D”层、Caffe 中的“卷积”层或 ONNX 中的“Conv”层都转换为 IR 中的“卷积”层。您可以在这里找到每个中间表示层本身的更深入的数据[。](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_IRLayersCatalogSpec.html)

OpenVINO 支持的框架:-

*   张量流
*   咖啡
*   MXNet
*   ONNX(PyTorch 和 Apple ML)
*   卡尔迪

# 配置模型优化器

要使用模型优化器，您需要配置它，配置模型优化器非常简单，可以在命令提示符/终端中完成。

要配置模型优化器，请遵循以下步骤(在命令提示符/终端中键入命令):-

1.  转到 Openvino 目录:-

对于 Linux:- `cd opt/intel/openvino`

对于 Windows:- `cd C:/Program Files (x86)/IntelSWTools/openvino`

我在上面的命令中使用了默认的安装目录，如果您的安装目录不同，那么请导航到适当的目录。

2.转到 install _ prerequitites 目录:-

`cd deployment_tools/model_optimizer/install_prerequisites`

3.运行`install_prerequisites`文件

对于 Windows:- `install_prerequisites.bat`

对于 Linux:- `install_prerequisites.sh`

如果您想要为特定的框架配置模型，那么运行以下命令:-

**张量流** :-

Windows:- `install_prerequisites_tf.bat`

Linux:- `install_prerequisites_tf.sh`

**咖啡** :-

Windows:- `install_prerequisites_caffe.bat`

Linux:- `install_prerequisites_caffe.sh`

**MXNet** :-

视窗:- `install_prerequisites_mxnet.bat`

Linux:- `install_prerequisites_mxnet.sh`

**ONNX** :-

视窗:- `install_prerequisites_onnx.bat`

Linux:- `install_prerequisites_onnx.sh`

卡尔迪 :-

Windows:- `install_prerequisites_kaldi.bat`

Linux:- `install_prerequisites_kaldi.sh`

# 转换为中间表示

在成功配置了模型优化器之后，我们现在就可以使用模型优化器了。在本文中，我将向您展示如何将 ONNX、Caffe 和 TensorFlow 转换为中间表示。ONNX 和 Caffe 的转换非常简单，但是 Tensorflow 模型的转换有点复杂。

**转换 ONNX 型号**

OpenVINO 不直接支持 PyTorch 相反，PyTorch 模型被转换成 ONNX 格式，然后由模型优化器转换成中间表示。

我将下载和转换“盗梦空间 _V1”。你可以从这个[链接](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html)找到其他型号。

下载“盗梦空间 _V1”后，解压缩文件并提取到你想要的位置。在“inception_v1”目录中，您会找到“model.onnx”文件。我们需要将该文件提供给模型优化器。

遵循以下步骤:-

1.  打开命令提示符/终端，将当前工作目录更改为“model.onnx”文件所在的位置
2.  运行以下命令:-

```
python opt/intel/opevino/deployment_tools/model_optimizer/mo.py --input_model model.onnx
```

*   **- input_model** - >取我们想要转换的模型。

上面的命令是在 Linux 中运行的，我使用了默认的安装目录，如果你的安装目录不同，那么使用适当的路径到“mo.py”。

```
python <installation_directory>/opevino/deployment_tools/model_optimizer/mo.py --input_model model.onnx
```

成功运行该命令后，您将会收到。xml“和”。bin”文件。

**转换咖啡模型**

Caffe 模型的转换过程非常简单，类似于 ONNX 模型。不同之处在于，对于 Caffe 模型，模型优化器采用了一些特定于 Caffe 模型的附加参数。您可以在[文档](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html#Convert_From_Caffe)中找到更多详细信息。

我将下载并转换 [SqueezeNet V1.1](https://github.com/DeepScale/SqueezeNet) 模型。

遵循以下步骤:-

1.  打开命令提示符/终端，将当前工作目录更改为“squeezenet_v1.1.caffemodel”文件所在的位置
2.  运行以下命令:-

```
python opt/intel/opevino/deployment_tools/model_optimizer/mo.py --input_model squeezenet_v1.1.caffemodel --input_proto deploy.prototxt
```

*   **- input_model →** 获取我们想要转换的模型。
*   **- input_proto →** 将包含拓扑结构和层属性的文件(deploy.prototxt)作为输入。

如果文件名为"。caffemodel“和”。prototxt”相同，则不需要参数“— input_proto”。

成功运行该命令后，您将会收到。xml“和”。bin”文件。

**转换张量流模型**

开放模型动物园中的 TensorFlow 模型采用冻结和解冻格式。TensorFlow 中的某些模型可能已经为您冻结了。您可以冻结模型，也可以使用中的单独说明来转换未冻结的模型。

您可以使用以下代码来冻结一个未冻结的模型。

```
import tensorflow as tffrom tensorflow.python.framework import graph_iofrozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["name_of_the_output_node"])graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
```

*   **sess →** 是定义网络拓扑的 TensorFlow* Session 对象的实例。
*   **["输出节点名称"] →** 是图中输出节点名称的列表；“冻结”图将仅包括直接或间接用于计算给定输出节点的原始“sess.graph_def”中的那些节点。
*   **。/ →** 是应该生成推理图文件的目录。
*   **推理图. pb** →是生成的推理图文件的名称。
*   **as_text** →指定生成的文件是人类可读的文本格式还是二进制格式。

我将下载和转换更快的 R-CNN 盗梦空间 V2 可可模型。你可以从这个[链接](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)找到其他型号。

下载“更快的 R-CNN 盗梦空间 V2 可可”后，解压缩文件并将其提取到您想要的位置。在“faster _ rcnn _ inception _ v2 _ coco _ 2018 _ 01 _ 28”目录里面，你会找到“frozen_inference_graph.pb”文件。我们需要将该文件提供给模型优化器。

遵循以下步骤:-

1.  打开命令提示符/终端，将当前工作目录更改为“freeze _ inference _ graph . Pb”文件所在的位置
2.  运行以下命令:-

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

上面的命令是在 Linux 中运行的，我使用了默认的安装目录，如果你的安装目录不同，那么使用适当的路径到“mo.py”。

*   **- input_model →** 获取我们想要转换的模型。
*   **-tensor flow _ Object _ Detection _ api _ pipeline**→用于生成借助对象检测 API 创建的模型的管道配置文件的路径。
*   **-reverse _ input _ channels**→TF 模型动物园模型是在 RGB(红绿蓝)图像上训练的，而 OpenCV 通常加载为 BGR(蓝绿红)。
*   **-tensor flow _ use _ custom _ operations _ config**→使用带有自定义操作描述的配置文件。

成功运行该命令后，您将会收到。xml“和”。bin”文件。

非常感谢您阅读这篇文章，我希望现在您已经对模型优化器有了正确的理解。