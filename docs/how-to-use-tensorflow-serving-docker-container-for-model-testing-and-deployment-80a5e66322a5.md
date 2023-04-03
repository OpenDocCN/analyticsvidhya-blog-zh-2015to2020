# 如何使用 tensor flow Serving docker 容器进行模型测试和部署

> 原文：<https://medium.com/analytics-vidhya/how-to-use-tensorflow-serving-docker-container-for-model-testing-and-deployment-80a5e66322a5?source=collection_archive---------0----------------------->

机器学习是一个涉及大量实验和研究的迭代过程。最终投产的车型只带来价值。将模型部署到产品中，同时确保它的可伸缩性并不是一件容易的事情。微服务是实现可伸缩性和健壮性的技术。数据科学家可以利用这种技术来实现为服务而部署的模型的可伸缩性、健壮性和可维护性。

除此之外，对要部署的模型进行 A/B 测试可能是重要的或者是一个好主意。有时，数据科学家希望部署使用不同超参数训练的同一模型的多个版本，以获得关于最终用户体验的一些反馈。

大多数时候，研究模型的数据科学家不能立即访问部署环境，或者发现编写定制的 REST API 来测试新训练的模型非常耗时。

如果以上任何一个任务让你担心，那么谷歌有一个现成的解决方案叫做 *Tensorflow Serving* 。

![](img/0b27cba32e2f29c92eed412f317bd9ce.png)

# 什么是 *Tensorflow 服务*？

根据谷歌的说法， *Tensorflow Serving* 是一个灵活、高性能的机器学习模型服务系统。它用于部署和服务机器学习模型。它可以同时为同一型号的多个版本提供服务。它对 Tensorflow 模型有现成的支持。

这是一个非常基础的教程，使用 *Tensorflow Serving* 在本地快速测试你新训练的 Tensorflow 模型。此外，您可以使用相同的技术轻松地在生产中部署模型。如果用在类似于 *Kubernetes* 的容器编排系统中，它可以开箱即用。

# 码头工人！这是什么？

在继续下一步之前，您应该熟悉 Docker。 *Docker* 是一个开放平台，开发人员和系统管理员可以在笔记本电脑、数据中心虚拟机或云上构建、发布和运行分布式应用。这里有一个非常好的教程，介绍了 Docker 的基础知识。

[](https://docker-curriculum.com/) [## 面向初学者的 Docker 教程

### 学习使用 Docker 轻松构建和部署您的分布式应用程序到云中，Docker 由…

docker-curriculum.com](https://docker-curriculum.com/) 

通常，为了开发一个 RESTful webservice，并为使用 Tensorflow 训练的模型提供一个预测端点，需要使用您选择的框架(如 Flask)用 Python 编写一个 webservice。下一步是使用 Alpine Linux 或 Ubuntu 基础映像编写 Dockerfile，安装必要的库等，并构建要部署的映像。

所有这些步骤都已经在一个 *Tensorflow 服务* Docker 映像中完成。最终用户只需使用新训练的模型部署容器，他/她就可以开箱即用地获得用于预测的 REST 端点。在以下步骤中，我将简要说明如何使用 *Tensorflow 服务* Docker 图像。

# 在 Tensorflow 服务容器中构建和部署模型的步骤

我已经使用 Keras 和 Tensorflow 训练并保存了一个模型。以下步骤可用于保存使用 Keras 训练的模型，以部署到 Tensorflow 服务容器中。

```
from keras.layers.core import K
from tensorflow.python.saved_model import builder as saved_model_builder

 model = Sequential()
 …..
 …
 model.compile(…)

 K.set_learning_phase(0)
 config = model.get_config()
 weights = model.get_weights()
 new_model = Sequential.from_config(config)
 new_model.set_weights(weights)

 builder = saved_model_builder.SavedModelBuilder(export_path)
 signature = predict_signature_def(
 inputs={'input': new_model.inputs[0]},
 outputs={'output': new_model.outputs[0]})

 with K.get_session() as sess:

 builder.add_meta_graph_and_variables(
 sess=sess,
 tags=[tag_constants.SERVING],
 clear_devices = True,
 signature_def_map={
 signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
 )
 builder.save()
```

训练过程完成后，会在导出路径中创建一个名为“saved_model.pb”的文件和一个“variables”目录。变量目录中的文件是“variables.data-00000-of-00001”和“variables.index”。

在进行下一步之前，Docker 应该安装在您的系统上。

拉最新的 *Tensorflow 发球*的 docker 图片。这将提取安装了 *Tensorflow Serving* 的最小 docker 映像。

```
docker pull tensorflow/serving
```

如果您想了解 docker 容器中发生的事情，docker 文件在这里:

[](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile) [## 张量流/服务

### 面向机器学习模型的灵活、高性能服务系统——tensor flow/serving

github.com](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile) 

它公开端口 8501 用于 REST。现在，只需启动 docker 容器，使用保存模型的路径挂载卷。这里很重要的一点是，训练后保存的模型应该放在一个有编号的目录中，例如 00000123。这允许 Tensorflow 在部署时创建模型的一个版本，以便新模型可以有不同的版本。

```
docker run -p 8501:8501 -v <path to model parent directory>:/models/<model_name> -e MODEL_NAME=<model_name> -t tensorflow/serving &
```

该命令将启动一个 docker 容器，部署模型，并使 REST 端点可用于获取预测。

使用 curl 测试模型是否正确部署:

```
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
  -X POST http://localhost:8501/v1/models/<model_name>:predict
```

这就是在您的本地系统中快速测试新训练的模型所需的全部内容。为了有一个合适的测试环境，使用将输入数据转换为模型输入向量所需的预处理步骤，并在 python 脚本中轻松测试多个值。

希望这将帮助人们快速测试他们的模型的健全性等。在继续云部署之前。