# 英特尔 OpenVINO:推理引擎

> 原文：<https://medium.com/analytics-vidhya/intel-openvino-inference-engine-7ba5076dc6e0?source=collection_archive---------9----------------------->

在我之前的文章中，我已经讨论了 OpenVINO 工具包的[基础和 OpenVINO 的](/swlh/introduction-to-intel-openvino-toolkit-5f98dbb30ffb)[模型优化器](/analytics-vidhya/intel-openvino-model-optimizer-e381affa458c)。在这篇文章中，我们将探索:-

*   什么是推理机？
*   支持的设备
*   向推理引擎提供中间表示(IR)
*   推理请求
*   处理输出

![](img/48edc9a7968773f116c63b6d69d82c5b.png)

# 什么是推理机？

推理引擎，顾名思义，在模型上运行实际的推理。它仅适用于来自模型优化器的中间表示(IR)或已经以 IR 格式呈现的英特尔预训练模型。

与模型优化器一样，它根据模型的大小和复杂性提供改进，以提高内存和计算时间，推理引擎提供基于硬件的优化，以进一步改进模型。

推理机本身实际上是在 C++中构建的，导致整体更快的操作；然而，在 Python 代码中利用内置的 Python 包装器与之交互是非常常见的。

# 支持的设备

推理引擎支持的设备都是英特尔硬件:-

*   中央处理器
*   图形处理单元
*   神经计算棒
*   现场可编程门阵列

大多数情况下，一个设备上的操作将与其他支持的设备相同，但有时，当使用推理引擎时，某些硬件不支持某些层(不支持的层)，在这种情况下，有一些可用的扩展可以添加支持。我们将在本文后面讨论它们。

# 向推理引擎提供一个 IR

推理引擎有两个类:-

*   网络
*   IECore

**网络**

这个类读取中间表示(。xml &。bin 文件)并加载模型。

**类属性**

*   名称→加载网络的名称。
*   输入→包含网络模型所需输入的字典。
*   输出→包含网络模型输出的字典。
*   batch_size →网络的批量大小。
*   精度→网络的精度(INT8、FP16、FP32)
*   layers →返回字典，该字典按拓扑顺序将网络图层名称映射到包含图层属性的 IENetLayer 对象。
*   stats →返回 LayersStatsMap 对象，该对象包含将网络层名称映射到由 LayerStats 对象表示的校准统计数据的字典。

**__init__()**

它是类构造函数。它需要两个参数:-

*   模型→通往。xml 文件。
*   权重→通往。bin 文件。

返回 IENetwork 类的实例。

**成员功能**

**from_ir()**

从读取模型。xml 和。IR 的 bin 文件。它需要两个参数:-

*   模型:-的路径。IR 的 xml 文件
*   权重:-路径到。IR 的 bin 文件

返回 IENetwork 类的实例。

注意:您可以使用 IENetwork 类构造函数来代替 from_ir()

**重塑()**

重塑网络以更改空间维度、批量大小或任何维度。它使用一个参数:-

*   input_shapes →将输入图层名称映射到具有目标形状的元组的字典

注意:使用此方法之前，请确保目标形状适用于网络。将网络形状更改为任意值可能会导致不可预测的行为。

**连载()**

序列化网络并将其存储在文件中。它需要两个参数:-

*   path_to_xml →存储序列化模型的文件路径。
*   path_to_bin →存储序列化重量的文件路径

**IECore**

这是一个 Python 包装类，用于推理引擎。

**类属性**

*   available_devices **→** 设备返回为[CPU，FPGA.0，FPGA.1，MYRIAD]。如果某个特定类型的设备不止一个，则会列出所有设备，后面跟一个点和一个数字。

**成员功能**

它有许多成员函数，但是我将集中讨论三个主要函数

**load_network()**

将从中间表示(IR)读取的网络加载到具有指定设备名称的插件，并创建 I network 类的 ExecutableNetwork 对象。

它需要四个参数:-

*   网络→有效的 IENetwork 实例。
*   设备名称→目标插件的设备名称。
*   config →插件配置键及其值的字典(可选)。
*   num_requests →要创建的推断请求的正整数值。推断请求的数量受到设备能力的限制。

返回 ExecutableNetwork 对象。

**add_extension()**

将扩展库加载到具有指定设备名称的插件中。

它需要两个参数:-

*   extension_path →要加载到插件的扩展库文件的路径。
*   设备名称→要加载扩展的插件的设备名称。

**查询 _ 网络()**

使用指定的设备名称查询插件，当前配置支持哪些网络层。

它需要三个参数:-

*   网络→有效的 IENetwork 实例。
*   设备名称→目标插件的设备名称。
*   config →插件配置键及其值的字典(可选)。

返回支持它们的字典映射层和设备名称

**将模型(在 IR 中)加载到 IE**

我们将首先从导入所需的库开始(我将使用 Python)

```
**from** openvino.inference_engine **import** IENetwork 
**from** openvino.inference_engine **import** IECore
```

让我们定义一个函数来加载模型。

```
**def** load_IR_to_IE(model_xml):
 ### Load the Inference Engine API
    plugin = IECore() ### Loading the IR files to IENetwork class
    model_bin = model_xml[:-3]+"bin" 
    network = IENetwork(model=model_xml, weights=model_bin)
```

**检查无支撑层**

如上所述，即使在成功转换到 IR 后，仍有一些硬件设备不支持某些层，我们有一些扩展可以添加支持。

当在 CPU 上使用推理引擎时，可能会有 CPU 不支持的某些层，在这种情况下，我们可以添加 CPU 扩展来支持其他层。

我将使用 IECore 类的“query_network()”来获取推理引擎支持的层列表。然后，您可以遍历您创建的 IENetwork 中的层，并检查它们是否在支持的层列表中。如果不支持某个层，CPU 扩展可能会有所帮助。

“device_name”参数只是一个字符串，用于表示设备“CPU”、“GPU”、“FPGA”或“MYRIAD”(适用于 Neural Compute Stick)。

让我们添加 CPU 扩展并检查不支持的层

```
 ### Defining CPU Extension path
    CPU_EXT_PATH=      "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/ libcpu_extension_sse4.so" ### Adding CPU Extension
    plugin.add_extension(CPU_EXT_PATH,"CPU")
```

虽然它们应该仍然在同一个位置，但是它们在操作系统上还是有一些不同(我在 Linux 上工作)。如果您导航到您的 OpenVINO 安装目录，然后是 deployment_tools、inference_engine、lib、intel64。

```
 ### Get the supported layers of the network
    supported_layers = plugin.query_network(network=network, device_name="CPU") ### Finding unsupported layers
    unsupported_layers = [l **for** l **in** network.layers.keys() **if** l **not** **in** supported_layers] ### Checking for unsupported layers
    **if** len(unsupported_layers) != 0:
        print("Unsupported layers found")
        print(unsupported_layers)
        exit(1)
```

上面的代码检查不支持的层的存在。为了简单起见，我们来分解一下上面的代码。

*   我已经使用了 query_network()，它是 IECore 类(上面提到的)的成员函数，用来获取支持的层的列表。
*   “network”是 I network 类的一个对象，它有一个名为“layers”的属性(如上所述)，它返回一个字典，其中包含每个支持的层的名称(作为键)及其属性(作为值)。利用这一点，我们找出不支持的层(如果存在)。
*   最后，如果存在任何不支持的层。我们显示消息和不支持的层，然后退出。

让我们加载网络

```
 ### Loading the network
    executable_net = plugin.load_network(network,"CPU") print("Network succesfully loaded into the Inference Engine") return executable_net
```

注意:-只有到 2019R3 版本的 OpenVINO 工具包才需要 CPU 扩展。在 2020R1(以及未来可能的更新)中，CPU 扩展不再需要添加到推理引擎中

# **推理请求**

在将 IENetwork 加载到 IECore 中之后，您将获得一个 ExecutableNetwok，这是您将向其发送推理请求的对象。

有两种类型的推理请求

*   同步的
*   异步的

**同步**

在同步推理的情况下，系统将等待并保持空闲，直到返回推理响应(阻塞主线程)。在这种情况下，一次只处理一个帧，并且在当前帧的推断完成之前不能收集下一个帧。

对于同步推理，我们使用“infer()”

**推断()**

它需要一个参数:-

*   输入→将输入图层名称映射到 numpy.ndarray 对象的字典，这些对象的形状与图层的输入数据一致

返回一个字典，该字典将输出图层名称映射到带有图层输出数据的 numpy.ndarray 对象

**异步**

正如您可能已经猜到的那样，在异步推理的情况下，如果对特定请求的响应花费了很长时间，那么您不会暂停，而是在当前流程执行的同时继续下一个流程。与同步推理相比，异步推理确保了更快的推理。

当主线程在同步模式下被阻塞时，异步模式不会阻塞主线程。因此，您可以发送一个帧进行推断，同时仍然收集和预处理下一个帧。您可以利用“等待”过程来等待推理结果可用。

对于异步推理，我们使用“start_async()”

**start_async()**

接受两个参数:-

*   request_id →开始推断的推断请求的索引。
*   输入→将输入图层名称映射到 numpy.ndarray 对象的字典，这些对象的形状与图层的输入数据一致

让我们实现同步和异步推理

```
**def** synchronous_inference(executable_net, image): ### Get the input blob for the inference request
    input_blob = next(iter(executable_net.inputs)) ### Perform Synchronous Inference
    result = executable_net.infer(inputs = {input_blob: image})
    return result
```

上面的代码显示了同步推理，input_blob 用作字典的键，字典作为参数提供给 infer()。infer()返回结果，该结果由函数返回。

```
**def** asynchronous_inference(executable_net, request_id=0, image): ### Get the input blob for the inference request
    input_blob = next(iter(executable_net.inputs)) ### Perform asynchronous inference
    executable_net.start_async(request_id=request_id, inputs={input_blob: image}) **while True**:
        status = executable_net.requests[request_id].wait(-1)
        **if** status == 0:
            **break**
        **else**:
            time.sleep(1)
    **return** executable_net
```

上面的代码展示了异步推理，这里我们提供了推理请求的 request_id(在多个推理的情况下)。如上所述，异步推理使用 wait()等待推理结果可用，如果结果可用(status=0)，则它退出循环，否则它等待 1 秒。如果我们调用 wait(0)，它将立即返回状态，即使处理没有完成。但是如果我们调用 wait(-1)，它将等待进程完成。

因此，异步推理不会像同步推理那样阻塞主线程。

# 处理输出

让我们看看如何从异步推理请求中提取结果。

```
**def** asynchronous_inference(executable_net, request_id=0, image): ### Get the output_blob
    output_blob = next(iter(executable_net.outputs)) ### Get the status
    status = executable_net.requests[request_id].wait(-1)

    **if** status == 0:
        result = executable_net.requests[request_id].outputs[output_blob]
        return result
```

如上所述,“输出”是保存网络模型输出的字典。“output_blob”充当访问特定输出的键。

非常感谢您阅读这篇文章。我希望到现在为止，您已经对推理引擎有了正确的理解。