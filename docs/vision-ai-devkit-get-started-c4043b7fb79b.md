# 视觉人工智能开发套件—开始使用

> 原文：<https://medium.com/analytics-vidhya/vision-ai-devkit-get-started-c4043b7fb79b?source=collection_archive---------10----------------------->

最后，还有另一个 AI 开发平台，它具有完整的外形，并且可以在现实世界中进行测试。我的意思是，高通公司的 [Vision AI Devkit](https://azure.github.io/Vision-AI-DevKit-Pages/docs/projects/) 采用完整的外壳制成，可以随时安装，如果需要安装测试的话。我们不需要找到单独的外壳或机箱和安装支架。

对其他例子感兴趣的[这里的](https://azure.github.io/Vision-AI-DevKit-Pages/docs/projects/)是一个很好的起点。

ok 是时候开始做开发了。首先从[这里](https://www.arrow.com/en/products/eic-ms-vision-500/einfochips-limited)获取开发套件。

 [## 视觉人工智能开发工具包|由 eInfochips 有限公司制造的 Altek |未分类| Arrow.com

### 从视觉人工智能开发工具包开始，创建您自己的基于图像的应用程序，如门口或窗户…

www.arrow.com](https://www.arrow.com/en/products/eic-ms-vision-500/einfochips-limited) 

一旦你得到了开发工具包，现在开始你的人工智能开发。没必要是数据科学家的。对于基于视觉的用例，从定制视觉认知服务开始很容易。首先用 Azure IoT Hub 安装和配置 devkit。一旦开发套件启动并运行，连接一个 HDMI 监视器，看看您是否可以查看屏幕。默认模型将能够检测人和许多常见物体。

现在是时候建造一些东西了。如果你有任何想法，继续发展。如果没有，请不要担心，这里有很多社区项目和教程可以帮助你开始。以下是开始使用的链接。

[](https://microsoft.github.io/ai-at-edge/docs/hw_examples/) [## 结合人工智能模型和硬件的例子

### 这里有一个使用硬件和人工智能模型的经过验证和测试的产品示例列表。大多数例子…

microsoft.github.io](https://microsoft.github.io/ai-at-edge/docs/hw_examples/) 

基于自定义视觉的项目将非常容易开始。要做到这一点，决定你想检测什么，例如我想检测安全帽。现在去收集一些不同背景、不同人物和各种各样戴着安全帽的人的照片。例如，50 个图像可能是一个好的开始。登录 customvision.ai 网站。为对象检测模型创建一个精简模型项目。上传图片，然后浏览每张图片，画出边界框来突出安全帽。将边界框标记为安全帽。现在点击培训按钮，然后发布。一旦发布，就可以导出并下载为 Vision AI DevKit 模型，这将生成所有必要的 zip 文件。

下载压缩文件并上传到 blob 存储器。一旦获得 blob 文件 URI，进入 Azure 物联网中心，选择物联网边缘，你应该会看到视觉套件。选择设备并选择视觉样本模块，然后单击模块 twin 链接。滚动以查找 ModelZipUrl 在哪里，并更新从 blob 属性中复制的完整 URI。保存模块 twin 文件并等待几分钟，模型将被下载到 Vision AI Devkit 中。现在你可以得到一个安全帽，并测试，看看模型的表现如何。

模型的准确性全靠你来让它运转起来。有关其他项目和教程，请查看 Vision AI Devkit 网站。还有一个[工作场所安全](https://azure.github.io/Vision-AI-DevKit-Pages/docs/community_project02/)社区项目，你可以测试看看。也可以尝试其他社区项目。

我们希望您能对所有社区项目和产品提出反馈意见。如果你有新的想法和需要帮助，请联系我们。使用网站中的联系我们。

如果您对构建智能边缘解决方案的基本原则感兴趣，您可以访问 [AI@Edge](https://microsoft.github.io/ai-at-edge/) 社区页面，了解更多硬件选项