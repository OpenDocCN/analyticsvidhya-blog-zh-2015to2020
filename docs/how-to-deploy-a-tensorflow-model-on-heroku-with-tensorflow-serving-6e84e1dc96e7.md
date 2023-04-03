# 如何在 Heroku 上部署 Tensorflow 模型并提供 Tensorflow 服务

> 原文：<https://medium.com/analytics-vidhya/how-to-deploy-a-tensorflow-model-on-heroku-with-tensorflow-serving-6e84e1dc96e7?source=collection_archive---------2----------------------->

![](img/b8e94d44e8b749c3118b7ef054e4f4d9.png)

图片来自 [Adobe 博客](https://xd.adobe.com/ideas/principles/emerging-technology/what-is-computer-vision-how-does-it-work/)

在花费数分钟或数小时玩转所有可用的精彩示例后，例如在[谷歌人工智能中心](https://aihub.cloud.google.com/s?category=notebook)，人们可能想要在线部署一个或另一个模型。

这篇文章介绍了一种用 [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) 和 [Heroku](https://www.heroku.com) 实现的**快速、最优和简洁**的方式。

# 介绍

在使用例如 [Google colab](https://colab.research.google.com) 在单个笔记本电脑上训练和测试单个模型与将模型部署到可以处理更新、批处理和异步预测等的生产环境之间存在差距。

幸运的是 **Google 已经公开发布了自己的框架来管理一个模型的整个生命周期**从数据存储到服务到日志等等。对于任何数据科学家或软件工程师来说，阅读 [Tensorflow Extended](https://www.tensorflow.org/tfx) 文档都是有价值的，因为他们正在寻找关于部署模型和实际应用程序的信息。

这个简单的 [REST API 示例](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)举例说明了如何训练一个简单的分类器，并最终使用服务于 Rest API 的 [Tensorflow 进行推理。](https://www.tensorflow.org/tfx/guide/serving)

这篇简短的博文进一步展示了如何在 Heroku 上部署这个模型，这样它就可以在线使用，并且可以被另一个 API 使用。

# 关于为模特服务

为模型服务就是使用它来进行某种预测:它应该被请求输入并返回一个有希望相关的答案。从服务的角度来看，模型是一个应该稳定返回输出的黑盒。

换句话说，当考虑为模型服务时，您不应该必须重新构建它。服务是模型不可知的:如果你服务一个计算机视觉分类器，它的目标是接收**要分类的原始图片**，并返回每个标签的分数。也就是说，裁剪、预处理等。步骤应该在为服务而构建模型时封装到模型中，即在培训结束时封装。

事实上，在生产思维状态下，每个经过训练的模型都有可能成为你问题的 SOTA。来自服务期间要接收的原始数据和最终预测的所有内容都应该存储在同一个对象中，以便任何人(尤其不是训练模型的数据科学家)都可以安全地使用它，而不会出现常见错误，如像素规范化、错误裁剪等。

# 使用张量流

[Tensorflow](https://www.tensorflow.org/) 库公开了 [saved_model](https://www.tensorflow.org/api_docs/python/tf/saved_model) API，这是专门为将模型打包成二进制跨平台格式而设计的，以后可以在任何地方使用而不会出现问题。`signatures`参数允许从相应的 REST API 定义几个要在模型上执行的路线和操作。

来自[Keras-FewShotLearning repo](https://github.com/few-shot-learning/Keras-FewShotLearning)的笔记本 [build for serving](https://github.com/few-shot-learning/Keras-FewShotLearning/blob/master/notebooks/build_siamese_model_for_serving.py) 是一个很好的例子，说明了如何使用`@tf.function`来创建路线(签名),然后可以使用 tensorflow serving API 轻松调用这些路线。

例如，给定训练期间使用的预处理:

人们可以加上下面的`@tf.function`:

并导出模型，如下所示:

这样，当接收完整的 base64 编码图像时，默认签名将返回每个类的格式化分数，而调用“预处理”签名将仅返回预处理图像。

来自[Keras-FewShotLearning repo](https://github.com/few-shot-learning/Keras-FewShotLearning)的 [request_served_model.py 笔记本](https://github.com/few-shot-learning/Keras-FewShotLearning/blob/master/notebooks/request_served_model.py)显示了如何使用 docker 运行 tensorflow 服务以及如何请求不同的签名，例如:

# 赫罗库

到目前为止，我们已经在本地运行了该模型。服务是从 docker 容器内完成的(即从 tensor flow/服务图像)。使用任何容器编排系统，如 [docker-compose](https://docs.docker.com/compose/) 或 [Kubernetes](https://en.wikipedia.org/wiki/Kubernetes) 将允许您清楚地将 tensorflow 提供的 ml 模型与应用程序或任何其他服务分开。你将从谷歌人的所有优秀工作中受益，例如，一旦新版本被推送到目标目录，就热重装模型:将新模型放入`classifier/2`将自动用新模型重装服务 API。

然而，如果你想在 Heroku 上部署这个容器，你将面临最后一个小困难，我现在要减轻这个困难。Tensorflow 服务通过端口 8501 提供 Rest API，但是 Heroku 在运行 dyno 时会分配一个随机端口。

因此，必须更新 tf-serving 命令的默认端口。自定义 Dockerfile 文件如下:

考虑到`$PORT` env 变量，对入口点稍作修改:

瞧啊！你的模型现在可以在地球上的任何地方被请求。如果你想从一个静态网站调用它，你可能会面临一个 CORS 问题，但这是另一回事。

# 结论

我在本教程中介绍了如何使用 Tensorflow 扩展框架来构建、部署和服务一个具有高效 API 的 tensorflow 模型(我猜是来自 Google)。我很想听听你关于使用 TFX 的其他好处，技巧等等。

别忘了关注我的更新和其他 tensorflow 相关文章！