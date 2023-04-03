# 用康达隐藏你的秘密钥匙

> 原文：<https://medium.com/analytics-vidhya/hide-your-secret-keys-with-conda-1d22e42b82ed?source=collection_archive---------7----------------------->

## 辅导的

## 使用 conda & Windows 在虚拟环境中隐藏密钥、id 和其他敏感信息的数据科学家指南。

![](img/b9d73a0f7abadfae937cc332a65f7b7a.png)

[杰森·D](https://unsplash.com/@jdphotomatography?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

我开始了一个专注于新冠肺炎数据的新数据项目。获取数据并存储它涉及到相当多的 API 键& IDs。作为一名深思熟虑的数据科学家，当我开始我的项目时，我创建了一个新的虚拟环境，就像我们都做的那样，对吗？无论如何——我将利用 AWS 和 Google 的 API，并决定我需要隐藏我的秘密密钥，因为这个项目将被发布到公共回购中。在那一点上，我意识到我以前没有这样做过——在上传我的代码之前，我会简单地删除或者用‘YOUR _ SECRET _ KEY _ HERE’交换密钥。不够好(见下文)。在这里，我们将介绍如何在 anaconda 虚拟环境中隐藏您的密钥，使用 Windows 的环境变量。

## 什么是密钥？

当我说“密钥”时，我指的是任何类型的服务帐户密钥、秘密访问密钥、访问 id、API 密钥…凭证，用于授权访问受限服务或区域，在本例中为 API。

![](img/32d2d987c2fbe7511b459087656d82da.png)

照片由[德米特里·德米德科](https://unsplash.com/@wildbook?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 为什么我们要隐藏它们？

嗯——不隐藏密钥可能会导致一些相当昂贵的问题，例如 2300 美元的比特币采矿 AWS 账单，或者像 Adobe [偶然在他们的安全博客上发布密钥](https://www.zdnet.com/article/adobe-accidentally-releases-private-pgp-key/)或优步试图[通过使用他们从 GitHub 获得的密钥之一来掩盖数据泄露](https://www.informationsecuritybuzz.com/expert-comments/uber-hack/)这样的尴尬时刻。看起来很多这样的密匙可以在像 GitHub 这样共享代码的回购网站上找到。坏演员是[使用机器人抓取 GitHub](https://www.cryptoglobe.com/latest/2020/05/hacker-steals-1200-worth-of-eth-using-github-bots/) 这些类型的密钥。不要试图记住代码中所有使用过该键的地方，最好从一开始就隐藏该键。

> 开发人员偶然地，而且经常是不知不觉地，在 GitHub 上一直共享凭证，从而暴露了他们的身份。虽然传统的安全控制对组织的安全仍然至关重要，但是如果能够访问私人信息的个人在一个可以被他人获取和滥用的地方暴露他们的帐户凭据，这是不好的。——Jeremiah Grossman，SentinalOne 的顾问

我发现有相当多的参考资料解释了如何向环境变量添加项目。对我来说，我见过的最常见的 Windows 选项不是很有条理，因为你想隐藏的所有项目都在一个地方。如果您有多个项目，这些项目可能会失去控制，并且随着环境变量列表的不断增长，对它们进行唯一命名可能会变得很麻烦。

**注意:**如果你使用的是 AWS 这样的云平台，最好让你的程序承担一个“角色”，因为这个角色将获得临时安全凭证，而不需要使用访问 id 或密钥。这是 AWS 用户的最佳实践，你可以在这里了解更多[。如果该选项不可用，则流程如下。](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)

还有其他创建虚拟环境的包，如 pipenv 和 virutalenv，但我们将通过一种方法来使用 conda 和 Windows 添加/隐藏这些键。你可以点击查看 conda [文档。](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

在这些步骤中，我们将创建一个新的虚拟环境—如果您已经有了一个环境，请跳至步骤 2。如果你不使用虚拟环境，你可能想！[了解它们是什么](https://www.datacamp.com/community/tutorials/virtual-environment-in-python)，为什么要使用它们，以及如何创建它们。

**下面是虚构的项目详情:**
要创建的新环境: *my_new_env* 批处理文件名:***envv-AWS creds . bat* **My ACCESS _ KEY _ ID**:' KLSNVEKWHLDYSK '
**My SECRET _ ACCESS _ KEY**= ' skdkjsnal 4489 kdsjl 49**

**首先，对于一个新项目，我们需要创建我们的新环境。——打开命令提示符并输入。**

```
conda create -n my_new_env
```

****提示|** 你可以在这里安装你需要的包，比如 python &的特定版本，使用`conda create -n myenv python=3.7`**

****2|进入您新创建的环境****

```
conda activate my_new_env
```

****3|定位并输入您的新虚拟环境的目录**
`%CONDA_PREFIX%`将显示您的新虚拟环境的位置。**

```
cd %CONDA_PREFIX%
```

****提示|** 您可以使用`cd %CONDA_PREFIX%`命令或者导航到您的 Anaconda 文件夹并导航到您的环境的存储位置。通常在`Anaconda\envs\`
下，现在让我们创建两个新目录，并在新目录下创建两个新文件。**

```
mkdir .\etc\conda\activate.d
mkdir .\etc\conda\deactivate.d
type NUL > .\etc\conda\activate.d\envv-awscreds.bat
type NUL > .\etc\conda\deactivate.d\envv-awscreds.bat
```

**第 1 & 2 行创建我们的新目录，第 3 & 4 行创建空的批处理文件，供 conda 在启动虚拟环境时运行。**

****提示|** 你可以随意命名这些批处理文件为`envv-awscreds.bat`。**

****4|导航至两个新创建的文件并编辑** 您可以使用记事本编辑这些文件。**

**编辑 `\etc\conda\activate.d\envv-awscreds.bat`文件**

```
set ACCESS_KEY_ID=KLSNVEKWHLDYSK
set SECRET_ACCESS_KEY=skdKJSNAL4489kdsjl49nkLKJJL
```

**注意| 我的键周围没有引号，赋值语句中也没有空格。**

**编辑 `\etc\conda\deactivate.d\envv-awscreds.bat`文件**

```
set ACCESS_KEY_ID = 
set SECRET_ACCESS_KEY = 
```

**很好——现在当我们运行 activate my_new_env 时，环境变量 *ACCESS_KEY_ID* 和 *SECRET_ACCESS_KEY* 将被设置为我们在文件中输入的变量。当我们停用环境时，变量将被删除。**

**停用环境，以便我们可以加载它，并仔细检查它的工作。**

```
conda deactivate
```

****5|仔细检查分配是否有效** 从命令提示符启动您的虚拟环境**

```
conda activate my_new_env
```

**输出:**

```
C:\>set ACCESS_KEY_ID=KLSNVEKWHLDYSK
C:\>set SECRET_ACCESS_KEY=skdKJSNAL4489kdsjl49nkLKJJL
(my_new_env) C:\>
```

**不错！我们看到它将我们的变量加载到环境中。让我们仔细检查一下是否可以用 python 来抓取它们。**

****6|将密钥加载到 python 脚本中**
要加载密钥，您需要使用 **os** 模块**

**启动 python**

```
(my_new_env) C:\> python
```

**导入操作系统模块，检索密钥并赋值给变量**

```
>>> import os
>>> ACCESS_KEY_ID = os.environ.get('ACCESS_KEY_ID')
>>> ACCESS_KEY_ID
```

**输出:**

```
'KLSNVEKWHLDYSK'
>>>
```

**呜哇！现在，您可以在 jupyter 笔记本或 python 文件中使用`ACCESS_KEY_ID`,而不是使用实际的密钥进行认证。没有未知的比特币给你挖矿！您仍然需要确保带有您的密钥的脚本文件不会被您的回购程序使用。跟踪一个批处理脚本要比扫描项目文件寻找任何键容易得多。**

**如果你认为你可能说漏了嘴，或者你在一个团队工作，想要再次确认没有机密的密钥、密码等。在你的回购中，有[种解决方案可以帮你走出](https://geekflare.com/github-credentials-scanner/)。**