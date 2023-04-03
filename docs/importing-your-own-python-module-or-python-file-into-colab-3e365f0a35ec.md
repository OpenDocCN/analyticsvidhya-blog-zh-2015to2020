# 将您自己的 Python 模块或 Python 文件导入 Colab

> 原文：<https://medium.com/analytics-vidhya/importing-your-own-python-module-or-python-file-into-colab-3e365f0a35ec?source=collection_archive---------0----------------------->

![](img/e661773238b85875287ec2f98df72c4d.png)

格伦·卡丽在 [Unsplash](https://unsplash.com/collections/8512483/solar-company?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

将您为自己的代码编写的 Python 模块或 Python 文件导入 Google Colaboratory。

我在搜索，有没有办法把**我的 Python 模块**或者 **Python 文件**代码( *my_module.py* 或者 *my_file.py* )导出到 **Google Colab lines** ？

首先，我像每个研究者一样在谷歌上做了研究。我看了看*栈溢出，Youtube，Medium* 。是的，我找到了很多代码，并尝试了它们。我将找到的代码应用到自己的代码中。我修改过了。可惜我没有成功。

当我放弃的时候，我修改了下面的代码，我发现这是一个解决方案，但不起作用。**是的，起作用了**。

以下步骤对我有效:

# 第一步

首先，你必须在 google colab 中挂载你的 google drive:
代码到下面，你的 Google drive 上的文件就是导入 google colab 里面的文件/包。

```
# Mount your google drive in google colab**from google.colab import drive
drive.mount('/content/drive')**
```

# 第二步

其次，使用 sys:

```
# Insert the directory**import sys
sys.path.insert(0,’/content/drive/My Drive/ColabNotebooks’)**
```

# 第三步

现在，您可以从该目录导入您的模块或文件。

```
# Import your module or file**import my_module**
```

也许有很多方法可以解决这个问题。如果你在其他方面给我建议，请在评论中提出。

还有，如果你想深度使用**Colab**，你绝对要去看看《 [***使用谷歌 Colab 进行深度学习***](https://neptune.ai/blog/how-to-use-google-colab-for-deep-learning-complete-tutorial) 》这篇文章。