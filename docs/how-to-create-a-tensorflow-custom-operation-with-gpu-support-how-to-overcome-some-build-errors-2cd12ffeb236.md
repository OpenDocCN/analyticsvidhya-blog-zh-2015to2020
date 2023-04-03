# 如何在 gpu 支持下创建 tensorflow 自定义操作(如何使用 docker 克服一些构建错误)

> 原文：<https://medium.com/analytics-vidhya/how-to-create-a-tensorflow-custom-operation-with-gpu-support-how-to-overcome-some-build-errors-2cd12ffeb236?source=collection_archive---------15----------------------->

![](img/b429a5d694452c34b69805c3d5e2ff2c.png)

[tensorflow 自定义操作](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwjjvdO7m-zmAhUtxoUKHYadBZEQjhx6BAgBEAI&url=https%3A%2F%2Fwww.tensorflow.org%2Ftutorials%2Fcustomization%2Fperformance&psig=AOvVaw1pPIe0xp8KTfM-NYpryVC7&ust=1578305262082985)

T 他的教程并不是构建定制操作的完整教程，但是它可以帮助你避免一些构建错误。Tensorflow 自定义操作有两个关于制作 tensorflow 操作的官方教程:[创建操作](https://www.tensorflow.org/guide/create_op)和[使用 docker 图像创建 pip 包](https://github.com/tensorflow/custom-op)。您可能会在这两个教程中找到所有问题的答案。不幸的是，事实并非如此。这些教程对于初学者来说并不像我想象的那么容易理解和详尽。事实上，使用 docker，用 c++、cuda 和 tensorflow 预制函数编程，手动链接或用 bazel 链接，最后用 python 测试是一项压力很大的任务。所以大概用@tf.function 对你的时间管理和心理健康会好很多。事实上，在完成这项任务之前，我每天都在做这件事，做了一个月。

![](img/827f432227bfa5d3863eb27fd6babdcc.png)![](img/b7fda74a8a885a9fa4ffd576004831bf.png)![](img/c803da107e07f0122a563c04b13b995f.png)

然而，如果您决定使用自定义操作，我可以提供一些帮助，告诉您应该遵循的一些命令。此外，如果您想进行 tensorflow 自定义操作，我强烈建议您使用 docker 图像。这个 **docker** 图像应该与您将在项目中使用的 tensorflow 兼容。例如，我使用了**tensor flow tensor flow:devel-GPU-py3**。

![](img/37009d859791d5d1520ca31050a272e6.png)

对我来说，主要的挑战是我在。cc 或. cu.cc 文件中。这些文件基本上来自**tensor flow/core/framework**和**tensor flow/core/common _ runtime/GPU**。事实上，问题是共享库文件成功创建，但导致了奇怪的错误，如 ***分段错误*** 和 ***找不到一些 cudnn 文件*** 总之，我会尽可能简单地解释我所做的步骤。

首先，从克隆 tensorflow 存储库开始。我遵循这个指南[从源代码](https://www.tensorflow.org/install/source) 来帮助我做到这一点。事实上，您只需要执行以下命令:

```
*git clone* [*https://github.com/tensorflow/tensorflow.git*](https://github.com/tensorflow/tensorflow.git)*cd tensorflow**git checkout r2.0 (you should take your time and know what branch you want to chose. Chose the branch that you will install in your python project.)*
```

准备好张量流分布之后，我们继续准备 docker 图像，我们将使用它来构建我们的操作。这个 docker 图像不是必须的，但是 tensorflow 的配置是一个非常微妙和烦人的事情。所以强烈推荐使用。你可能需要花几个小时来理解 [**docker**](https://www.tensorflow.org/install/docker) 是如何工作的，但这是一件值得学习的事情。如果你的操作支持 gpu，你需要安装 [**nvidia-docker**](https://github.com/NVIDIA/nvidia-docker) 。您必须选择一个与您选择的分支兼容的 docker 映像。

然后运行 docker 容器并将其挂载到 tensorflow 目录:

```
docker run — runtime=nvidia -it -w /tensorflow -v $PWD:/tensorflow -e HOST_PERMS=”$(id -u):$(id -g)” tensorflow/tensorflow:devel-gpu-py3 bash./configure
```

组织你的代码最好的方法就是在**/tensor flow/core/user _ ops/**中做一个目录，命名为:**tensor flow _ operation _ name**。代码可以这样组织:

```
├── tensorflow_operation_name # A GPU op│ ├── cc│ │ ├── kernels # op kernel implementation│ │ │ |── operation_name.h│ │ │ |── operation_name.cc│ │ │ └── operation_name.cu.cc # GPU kernel│ │ └── ops # op interface definition│ │ └── operation_name_ops.cc│ ├── python│ │ ├── ops│ │ │ ├── __init__.py│ │ │ ├── operation_name_ops.py # Load and extend the ops in python│ │ │ └── operation_name_ops_test.py # tests for ops│ │ └── __init__.py| |│ ├── BUILD # BUILD file
```

本教程不会考虑测试部分，也不会把操作变成 pip 可安装操作。所以所有的 __init__。py 和 operation_name_ops_test.py 都将保持为空，如果你真的想实现那些特性，可以参考开头提到的两个教程。

现在，您必须以一种有组织的方式将代码放入这些文件中。如果你不知道怎么做，那就花点时间从这里的[复制代码](https://github.com/tensorflow/custom-op)或者从这里的[复制代码](https://www.tensorflow.org/guide/create_op)。如果你不熟悉 c++、cuda，这个任务将会很难完成。问题是，你必须使用预定义的张量流代码添加到这两种编码语言中，这使得任务更加复杂。

成功完成所有这些步骤后，您现在需要完成构建文件。如果您的操作仅使用来自 **third_party/eigen3** 、**tensor flow/core/framework**、**tensor flow/core/platform**、 **tensorflow/core/util** 和 **tensorflow/core/lib** 的文件，此代码应该工作正常:

张量流 _ 操作 _ 名称/版本:

```
package(default_visibility = ["//visibility:public"])load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")tf_custom_op_library(name = 'python/ops/_operation_name_ops.so',srcs = ["cc/kernels/operation_name.h","cc/kernels/operation_name.cc","cc/ops/operation_name_ops.cc",],gpu_srcs = ["cc/kernels/operation_name.cu.cc", "cc/kernels/operation_name.h"],
)
```

如果您正在使用来自**tensor flow/core/common _ runtime**的其他文件或任何其他文件，您应该查找包含该文件的适当规则，并将其放入构建文件的 deps 中。比如我收录:*deps =[" @ cub _ archive//:cub "]*因为我用的是 cub 目录。再比如，我在包含了**tensor flow/core/common _ runtime/GPU**目录下的文件后，添加了*deps =["//tensor flow/core:GPU _ headers _ lib "]*。您必须理解，不包含您的依赖项可能会导致异常行为，例如分段错误和构建错误。如果你包含了多个文件，我建议你了解一下 [bazel](https://docs.bazel.build/versions/master/tutorial/cpp.html) 是如何工作的。

现在剩下的就是建造你的图书馆了。此构建指令适用于 gcc>=5。

```
bazel build — config=opt — config=cuda — cxxopt=”-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/core/user_ops/tensorflow_operation_name:python/ops/_operation_name_ops.so
```

如果没有编译错误，结果文件将会在某个 bazel 目录中构建。只需执行这个命令，就可以获得。所以文件:

```
find / -iname _operation_name_ops.so
```

找到您的文件后，您现在可以在 python 中使用*tensor flow . python . framework . load _ library(path _ to _)包含并使用它。然后测试你的操作，希望一切正常！*