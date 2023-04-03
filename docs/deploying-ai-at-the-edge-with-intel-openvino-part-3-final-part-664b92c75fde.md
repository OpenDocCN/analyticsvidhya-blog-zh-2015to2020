# 利用英特尔 OpenVINO 在边缘部署人工智能-第 3 部分(最后一部分)

> 原文：<https://medium.com/analytics-vidhya/deploying-ai-at-the-edge-with-intel-openvino-part-3-final-part-664b92c75fde?source=collection_archive---------12----------------------->

![](img/622b9a5635091fca341ba12cd76501a5.png)

约书亚·梅洛在 [Unsplash](https://unsplash.com/s/photos/engine?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

在我以前的帖子中，我已经介绍了 OpenVINO，描述了如何在 windows 计算机中安装它，如何处理输入和输出，以及如何在 model optimizer 中获取或准备模型。我们现在处于最后一步，用推理引擎执行推理。让我们开始吧。这篇文章讨论的主题是，

*   推理机
*   向推理机提供模型
*   检查不支持的层并使用 CPU 扩展(不推荐)
*   发送推理请求
*   处理输出
*   集成到应用程序中

# 推理机

推理引擎在模型上运行实际的推理。在[第 1 部分](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-1-51a09752fb4e)中，我们从 OpenVINO 模型动物园下载了一个预训练模型，在[第 2 部分](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-2-1f1a9faa514b)中，我们通过模型优化器将一些模型转换成 IR 格式。推理机只处理这种中间表示。在第 2 部分中，我们已经看到了模型优化器如何通过改进规模和复杂性来帮助优化模型。推理引擎提供进一步的基于硬件的优化，以确保使用尽可能少的硬件资源。因此，它有助于在物联网设备的边缘部署人工智能。

为了更好地与硬件通信，推理机建立在 C++之上。所以你可以直接在你的 C++应用中使用引擎。还有一个 python 包装器可以在 python 代码中使用引擎。在这篇文章中，我们将使用 python。

# 如何使用推理引擎

以下是从头到尾使用推理引擎的步骤，

> *馈送模型>检查任何不支持的层>发送推理请求>处理结果>与您的应用程序集成*

现在我们将看到每一步我们需要做的细节工作。在[第 1 部分](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-1-51a09752fb4e)中，我们创建了两个 python 文件，一个名为 *app.py* ，另一个名为 *inference.py* 。我们将从那里继续代码。

## 给模特喂食

![](img/f8339c4ffb3c8d001f4520294b4d784b.png)

在 [Unsplash](https://unsplash.com/s/photos/feed-bird?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上由 [Thibault Mokuenko](https://unsplash.com/@tmokuenko?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

我们需要使用来自*“open vino . inference _ engine”*库中的两个 python 类。分别是*I core*和*I network*。 [*IECore*](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html) 和 [*IENetwork*](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html) 的文档对于使用类中的不同方法会非常有帮助，所以请查阅它们。

*I network*保存从 IR 读取的关于模型网络的信息，并允许进一步修改网络。在必要的处理之后，它将网络提供给 IECore，后者创建一个[可执行网络](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html)。

我们将从在我们的*推论. py* 文件中导入必要的库开始。

```
from openvino.inference_engine import IECore
from openvino.inference_engine import IENetwork
```

现在我将创建一个名为*“load _ to _ IE”*的函数，它将接受一个参数 model(models 的位置 **)。xml* 文件),从中我们还将获得 **的位置。bin* 文件。此外，还有另一个名为“cpu_ext”的变量，我将在后面解释(例如，我正在使用我在[第 1 部分](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-1-51a09752fb4e)中使用的同一人脸检测模型)。

( ***更新:*** *自 OpenVINO 2020 起，不再需要 cpu 扩展。如果您使用的是更高版本，请忽略所有关于 cpu 扩展的内容。我把它放在这里，以防有人正在使用旧版本，需要它作为参考)*

```
cpu_ext_dll ="C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/inference_engine/bin/intel64/Release/cpu_extension_avx2.dll"
```

现在让我们定义这个函数。

```
def load_to_IE(model):
    # Getting the *.bin file location
    model_bin = model[:-3]+"bin" #Loading the Inference Engine API
    ie = IECore()

    #Loading IR files
    net = IENetwork(model=model_xml, weights = model_bin)
```

( ***更新:*** *像这样直接初始化 IENetwork 对象是不赞成的。使用下面的代码)*

```
def load_to_IE(model):
    ....
    net = ie.read_network(model=model, weights = model_bin)
```

## 检查不受支持的图层

( ***更新:*** *不适用于 OpenVINO 2020 及以后版本)*

![](img/b45b4e65ae7d515ce2ccb3bb8dd9b111.png)

兰迪·法特在 [Unsplash](https://unsplash.com/s/photos/puzzle-piece?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

即使在成功地将模型转换为 IR 后，CPU 可能仍然不支持某些层。在这种情况下，我们可以使用 CPU 扩展文件来支持推理引擎中那些不受支持的层。在不同的操作系统中，CPU 扩展文件的位置略有不同。例如，我找到了我的 CPU 扩展 **。该位置的 dll* 文件: *<安装 _ 目录>\ open vino _ 2019 . 3 . 379 \ deployment _ tools \ inference _ engine \ bin \ Intel 64 \ Release。*

文件命名为*“CPU _ extension _ av x2 . dll”*。在 linux 中有几个扩展文件，而在 Mac 中只有一个。

并非所有型号都需要 CPU 扩展。因此，首先我们将检查我们的模型是否需要它。继续“load_to_IE”函数内的代码。

```
# Listing all the layers and supported layers
    cpu_extension_needed = False
    network_layers = net.layers.keys()
    supported_layer_map = ie.query_network(network=net,device_name="CPU")
    supported_layers = supported_layer_map.keys()
```

让我解释一下代码。首先，我设置了一个标志，表明不需要 cpu 扩展。然后，我在“network_layers”变量中列出我们网络中所有层的名称。然后，我使用 IECore 类的*“query _ network”*方法，该方法返回当前配置中支持的层的字典。通过从字典中提取键，我创建了一个支持层的列表，并将其存储在“supported_layers”变量中。

现在，我将遍历当前网络中的所有层，并检查它们是否属于受支持的层列表。如果所有层都出现在受支持的层中，则不需要 cpu 扩展，否则，我们将把我们的标志设置为 false，并继续添加 CPU 扩展。在“load_to_IE”函数中继续下面的代码。

```
# Checking if CPU extension is needed   
    for layer in network_layers:
        if layer in supported_layers:
            pass
        else:
            cpu_extension_needed =True
            print("CPU extension needed")
            break
```

我们将使用 IECore 类的“add_extension”方法来添加扩展。继续相同函数中的代码。

```
# Adding CPU extension
    if cpu_extension_needed:
        ie.add_extension(extension_path=cpu_ext, device_name="CPU")
        print("CPU extension added")
    else:
        print("CPU extension not needed")
```

为了安全起见，我们将再次使用相同的代码来查看在添加 CPU 扩展之后，现在是否支持所有的层。如果现在所有的层都被支持，我们可以进入下一步。但如果没有，我们将退出程序。

```
#Getting the supported layers of the network  
    supported_layer_map = ie.query_network(network=net, device_name="CPU")
    supported_layers = supported_layer_map.keys()

# Checking for any unsupported layers, if yes, exit
    unsupported_layer_exists = False
    network_layers = net.layers.keys()
    for layer in network_layers:
        if layer in supported_layers:
            pass
        else:
            print(layer +' : Still Unsupported')
            unsupported_layer_exists = True
    if unsupported_layer_exists:
        print("Exiting the program.")
        exit(1)
```

既然我们确定所有的层都被支持，我们将把网络加载到我们的推理引擎中。

```
# Loading the network to the inference engine
    exec_net = ie.load_network(network=net, device_name="CPU")
    print("IR successfully loaded into Inference Engine.") return exec_net   #exec_net short for executable network
```

## 发送推理请求

![](img/908e8796c5e019b0ed7134eae8a1f889.png)

[布拉登·科拉姆](https://unsplash.com/@bradencollum?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/start?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

我们将把我们的推理请求发送到由我们的“load_to_IE”函数返回的可执行网络“exec_net”。有两种推断方法，*同步*和*异步。*在同步方法中，应用程序向引擎发送推理请求，除了等待推理请求完成之外什么也不做。另一方面，在异步方法中，当推理机执行推理时，应用程序可以继续其他工作。如果推理请求处理由于某种原因很慢，并且我们不希望我们的应用程序在推理完成时挂起，这是很有帮助的。因此，在异步方法中，当处理一帧上的推理请求时，应用程序可以继续收集下一帧并对其进行预处理，而不是在同步方法的情况下被冻结。

让我们定义两个函数。一个函数将执行同步推理，另一个函数将执行异步推理。名为*“sync _ inference”*的同步函数将可执行网络和预处理后的图像作为参数。名为*“async _ inference”*的异步推理函数将再接受一个额外的参数*“request _ id”*，我将其默认为 0，因为在本例中，我们将只发送一个请求，所以我们不需要为 id 而烦恼。当您使用这种方法使您的应用程序真正“异步”时(在我们的例子中，我们实际上是在等待 IE 完成推理，因为在我们简单的演示应用程序中没有其他事情要做)，您可能会在引擎完成对一个图像的推理之前，用不同的请求 id 一个接一个地输入几个图像。当引擎完成推理时，您将使用这些 id 提取相应的结果。因此，请确保在您的真实应用程序中为您的图像分配唯一的 id。你可以阅读[文档](https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#a314107a6b50f0ebf65751569973c760a)来了解更多关于异步方法的信息。处理这两种方法的输出是不同的。对于同步方法，我们可以直接返回结果。但是对于异步方法，我们将返回一个处理程序(由 *start_async* 方法返回)。

```
def sync_inference(exec_net, image):
    input_key = list(exec_net.input_info.keys())[0]
    output_key = list(exec_net.outputs.keys())[0]
    result = exec_net.infer({input_key: image})
    print('Sync inference successful')
    return result[output_key]def async_inference(exec_net, image, request_id=0):
    input_key = list(exec_net.input_info.keys())[0]
    return exec_net.start_async(request_id, inputs={input_blob: image})
```

(在第 1 部分中，我们首先打印从 *sync_inference* 函数返回的‘result’字典的键，然后硬编码该键以提取结果。我稍微修改了一下代码，这样我们就不再需要硬编码了。不同的型号对输入和输出键使用不同的名称。因此，通过以编程方式提取键名，我们使应用程序更适用于只有一个输入和一个输出的不同模型)

当异步推断完成时，可以使用请求 id 或处理程序提取结果，因此需要更多的处理。

## 处理输出

![](img/bd8532a3e12d9ed1aca9e2ef24d0e6af.png)

杰西·拉米雷斯在 [Unsplash](https://unsplash.com/s/photos/package?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

我们已经看到，同步方法直接给了我们推论的结果。在[第 1 部分](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-1-51a09752fb4e)中，我们已经使用了这个结果并处理了我们的输出来绘制边界框。当然，根据我们希望应用程序做什么，处理过程会有所不同。为了从异步方法中获得推理结果，我们将定义另一个函数，我命名为“ *get_async_output”。这个函数将接受一个参数。*“async _ inference”*返回的句柄。*

```
def get_async_output(async_handler):
    status = async_handler.wait()
    output_key = list(async_handler.output_blobs.keys())[0]
    result = async_handler.output_blobs[output_key].buffer
    print('Async inference successful')
    return result
```

*“wait”*函数返回加工状态。如果我们用参数 0 调用函数，它将立即返回状态，即使处理没有完成。如果您只是检查结果是否完整，这是很有用的，如果不完整，您可以继续其他过程。

## 与您的应用程序集成

![](img/0dd9de8b1a621f0ccf701e4cc2978d53.png)

克里斯·里德在 [Unsplash](https://unsplash.com/s/photos/code-python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

我们已经完成了我们的推理引擎。现在，我们将在非常简单的 *app.py* python 文件中使用我们的函数。首先，我们需要将刚刚在*推论. py* 文件中定义的函数导入到我们的 *app.py* 文件中。然后，在*“main”([第一部分](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-1-51a09752fb4e)中定义的*)中，我们将相应地调用这些函数，以便在我们的应用程序中使用它们。

```
from inference import preprocessing, load_to_IE, sync_inference, async_inference, get_input_shape, get_async_outputdef main(): #*..................................*# exec_net = load_to_IE(model, model_bin)

    # Synchronous method
    result = sync_inference(exec_net, image = preprocessed_image)
    #result = result['detection_out'], we are don't need this anymore #*..................................*# # or, Async method
    async_handler=async_inference(exec_net,image=preprocessed_image)
    result_async=get_async_output(async_handler)
```

## 奖金！

还记得吗，我们在第一部分中对所需的图像尺寸进行了硬编码？我们不需要再这样做了。在我们的 *"inference.py"* 文件中定义这个新函数。

```
def get_input_shape(net):
    input_key = list(net.input_info.keys())[0]
    input_shape = net.input_info[input_key].input_data.shape
    return input_shape
```

现在，导入 *app.py* 文件中的函数。并且，在*【main】*函数内添加这一行，而不是硬编码高度和宽度。

```
from inference import get_input_shapedef main():
    #*.............*#
    n, c, h, w = get_input_shape(exec_net)
    preprocessed_image = preprocessing(image, h, w)
```

我已经将完整的代码上传到了 github 库中。如果您发现代码的任何部分令人困惑，请检查完整的代码。

# 结论

英特尔 OpenVINO toolkit 拥有无限可能。巨大的模型动物园为开发人员提供了他们所需要的一切，以利用尖端的人工智能技术构建强大的边缘应用程序，而不必太担心培训。OpenVINO 还可以毫不费力地将人工智能带到你的低功耗设备中。即使在一个与英特尔 NCS 配对的小 raspberry pi 中，您也可以通过 OpenVINO 使用 AI 制作出令人惊叹的应用程序。你的想象力在这里是极限。我希望我的帖子已经帮助你开始使用 OpenVINO 并制作你的 edge 应用程序。我目前正在与一个名为 [Hessix](http://hessix.github.io) 的非营利组织合作，在那里我们应用深度学习来解决社会问题。我计划很快在那个项目中使用 OpenVINO。完成后，我可能会在那里发布一篇关于我如何使用 OpenVINO 的文章。所以下次见，享受编码吧。

(*链接到所有零件:* [*零件 0*](/@ahsan.shihab2/ai-at-the-edge-an-introduction-to-intel-openvino-toolkit-a0c0594a731c)*>*[*零件 1*](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-1-51a09752fb4e)*>*[*零件 2*](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-2-1f1a9faa514b)*>*[*零件 3*](/@ahsan.shihab2/deploying-ai-at-the-edge-with-intel-openvino-part-3-final-part-664b92c75fde) *)*