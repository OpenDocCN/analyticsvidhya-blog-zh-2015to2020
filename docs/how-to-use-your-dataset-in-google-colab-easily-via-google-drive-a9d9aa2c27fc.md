# 如何通过 Google Drive 轻松使用 Google Colab 中的数据集

> 原文：<https://medium.com/analytics-vidhya/how-to-use-your-dataset-in-google-colab-easily-via-google-drive-a9d9aa2c27fc?source=collection_archive---------19----------------------->

在 Google Colab 中使用数据集的简单方法

![](img/2c163edd5b2397a0aba7082ca216768f.png)

# 1.在 Google Drive 中创建一个文件夹

在这个例子中，我们在主*(我的驱动器)*目录中创建了一个名为“ **colab** 的文件夹。

![](img/6266432844e43d9d6ee0dc13ca02180b.png)

# 2.上传数据集

在这里，我们将数据集上传到“colab”文件夹。

![](img/23869a614f61af10b264ae4933679d44.png)

# 3.创建或打开 Colab 笔记本

在这里，我们创建了一个新的 colab 笔记本。

![](img/86eb1a0590dd2c380e37f558cb81cd67.png)

# 4.安装您的驱动器

使用以下代码安装驱动器:

```
# mount
from google.colab import drive
drive.mount('/content/drive')
```

![](img/f04396b8daaa21a7f6b375dceb21821a.png)

现在，我们需要获得授权码，所以我们点击给出的 **URL** 并完成授权步骤。

> 完成所有步骤后，我们最终得到了我们的授权码:

![](img/fbfbf31616714a2abf63019fa2ac61b8.png)

我们将授权码复制并粘贴到输入字段，然后按下 **Enter。**

![](img/e77e8e8cf8e7806fe5c5c186f72d0970.png)

> 注意:应该在每个会话中重复安装步骤。

# 5.从 Google Drive 读取数据集

我们已经创建了一个名为“ **colab** ”的文件夹，并将数据集上传到其中。因此，我们需要使用下面的路径来读取数据集:

```
"/content/drive/My Drive/colab/dataset.csv"
```

在这个例子中，我们使用 **pandas 读取数据集。**

![](img/dc63a43e1ffa9a4fee5c19731b3711f4.png)

# 6.笔记

*   **安装**步骤应在每个*会话*中重复。