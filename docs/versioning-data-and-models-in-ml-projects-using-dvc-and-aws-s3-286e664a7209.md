# 使用 DVC 和 AWS S3 对 ML 项目中的数据和模型进行版本控制

> 原文：<https://medium.com/analytics-vidhya/versioning-data-and-models-in-ml-projects-using-dvc-and-aws-s3-286e664a7209?source=collection_archive---------0----------------------->

在这篇博客中，我们将详细探讨如何使用 [DVC](https://dvc.org/doc/start) 对我们的数据和模型进行版本化。这个博客的代码可以在[这里](https://github.com/bhuvanakundumani/NER_tf_dvc_s3)找到。有关使用 Tensorflow2.2.0 对句子( [CoNLL-2003 数据集](https://www.clips.uantwerpen.be/conll2003/ner/))进行命名实体识别(NER)标记的模型训练的详细信息，请在此阅读博客。

## **什么是 DVC？**

数据版本控制，或 DVC，是一个数据和 ML 实验管理工具，非常类似于 Git。它帮助我们跟踪和保存数据和 ML 模型。DVC 将有关数据的信息保存在特殊的[元文件](https://dvc.org/doc/user-guide/dvc-files-and-directories)中，这些元文件替换了存储库中的数据。这些可以用常规的 Git 工作流(分支、拉请求等)来版本化。DVC 使用内置缓存来存储数据，并支持与 was s3、google drive、Microsoft azure、google cloud 等远程存储选项同步数据。

![](img/b56fc9446df8077858b5c88f07692537.png)

[*DVC 代码和数据流*](https://dvc.org/doc/use-cases/versioning-data-and-model-files)

我们将把数据文件夹中的数据、嵌入和模型输出存储在 S3 存储桶的 model_output 文件夹中。我已经创建了一个 s3 bucket s3://dvc-example。参考[链接](https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html)创建 S3 铲斗。

让我们从 dvc 开始吧。使用 pip 安装 dvc 包。要使用其他选项进行安装，请参考此[链接](https://dvc.org/doc/install)。还要安装 boto3、dvc[all]和 dvc[s3]包。

```
pip install dvc 
pip boto3 dvc[all] dvc[s3]
```

安装后，在 git 项目中初始化 dvc

```
dvc init
```

`.dvc/.gitignore`和`.dvc/config`被创建。提交此更改

```
git commit -m "Initialize DVC"
```

该项目的文件夹结构如下所示

```
├── data
│   ├── train.txt
│   ├── valid.txt
│   ├── test.txt
│
├── embeddings
│   ├── glove.6B.100d.txt
│   
├── model_output
│   ├── checkpoint
│   ├── embedding.pkl
│   ├── idx2Label.pkl
│   ├── model_weights.data-00000-of-00001
│   ├── model_weights.index
│   ├── word2Idx.pkl
```

我们将不得不跟踪数据、嵌入和模型输出目录。让我们首先用 dvc 跟踪数据文件夹。

```
dvc add data
```

DVC 将文件内容移动到缓存中。它还在数据文件夹中创建一个名为`data.dvc`的相应的`[.dvc](https://dvc.org/doc/user-guide/dvc-files-and-directories#dvc-files)`文件来跟踪文件，使用它的路径和散列来标识缓存的数据。它还将文件添加到数据文件夹`.gitignore`中，以防止它们被提交到 Git 存储库中。

提交。git 中的 dvc 文件。

```
git add data.dvc
git commit -m “add data”
```

类似地，使用 dvc 跟踪嵌入文件夹和 model_output 文件夹，使用如下所示的代码。

```
dvc add embeddings
git add embeddings.dvc
git commit -m “add embeddings”
dvc add model_output
git add model_output.dvc
git commit -m “add models”
```

让我们设置 s3 存储桶:S3://DVC——远程存储的例子。

```
dvc remote add -d remote s3://dvcexample/ner
```

该命令在 DVC 项目的[配置文件](https://dvc.org/doc/command-reference/config)中创建一个`remote`部分，并可选地将`core`部分中的一个*默认遥控器*分配给 myremote。

我们需要为 s3 存储桶设置 aws 访问密钥 id 和秘密访问密钥

```
dvc remote modify myremote access_key_id AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key AWS_SECRET_ACCESS_KEY
```

(注意:用您的 AWS_KEY_ID 替换 AWS_ACCESS_KEY_ID，用您的 AWS_SECRET_ACCESS_KEY 替换 AWS_SECRET_ACCESS_KEY。您不需要使用双引号)

关于您的远程存储器的信息被捕获并存储在`.dvc/config`中。它还有 s3 存储桶名、访问密钥 id 和秘密访问密钥。请确保在按下时删除访问密钥 id 和机密访问密钥。要 git 的配置文件。或者，您可以使用 credentialpath，这将在下面的使用 credentialpath 标题下详细介绍。

```
[core] remote = myremote['remote "myremote"'] url = s3://dvcexample/ner access_key_id = AWS_ACCESS_KEY_ID secret_access_key = AWS_SECRET_ACCESS_KEY
```

现在我们已经配置了 s3 远程存储，让我们推送数据、嵌入和模型输出文件。

```
dvc push
```

这三个目录中的所有文件现在都被推送到 s3 存储中。

让我们将`.dvc/config`文件提交给 git。确保从`.dvc/config`文件中删除 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY。酷！！！数据和模型存储在远程存储器中，dvc 帮助您对其进行版本化。

现在让我们将代码推送给 git。假设它在主支行，

```
git push origin master
```

# 使用凭据路径

让我们使用 aws cli 配置 aws。确保您已经安装了 awscli。

```
pip install awscli
```

然后在终端中输入`aws configure`。

当您输入此命令时，AWS CLI 会提示您输入四条信息。键入您的访问密钥 ID、秘密访问密钥、AWS 区域和输出格式。

```
$ **aws configure** 
AWS Access Key ID [None]: *AKIAIOSFODNN7EXAMPLE* 
AWS Secret Access Key [None]: *wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY* 
Default region name [None]: *us-west-2* 
Default output format [None]: *json*
```

AWS CLI 将这些信息存储在`credentials`文件中名为`default`的*配置文件*(一组设置)中。`**~/.aws/credentials**` 文件具有访问密钥 id 和秘密访问密钥。与下图非常相似。

```
[default] 
aws_access_key_id=AKIAIOSFODNN7EXAMPLE aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

既然您已经配置了 aws。您可以使用 credentialpath 来代替密钥。这将设置 s3 的访问。

```
dvc remote modify myremote credentialpath ~/.aws/credentials
```

这实际上编辑了如下所示的`.dvc/config`文件。

```
[core] remote = myremote['remote "myremote"'] url = s3://dvcexample/ner credentialpath = /Users/user1/.aws/credentials
```

# 获取数据和模型

团队需要做的就是 git 克隆存储库。设置虚拟环境。在 requirements.txt 文件中安装所有需要的文件。记住，`.dvc/config`没有 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY。用您在`.dvc/config`文件中的凭证替换 AWS_ACCESS_KEY_ID 和 AWS_SECRET_ACCESS_KEY。

```
dvc pull 
```

上面的命令提取所有的数据、嵌入和 model_output 文件。你都准备好了。

# 数据/模型的变化

假设您正在更改 train.py 中的一些超参数，并训练一个新模型。训练之后，model_output 有了新的模型文件。你需要做的就是

```
dvc add model_output
git commit model_output.dvc -m 'model updates'
dvc push
git push origin master
```

像`git checkout`一样，我们可以使用`dvc checkout`在不同版本的数据之间切换。

# 参考

[](https://towardsdatascience.com/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-7cb49c229fe0) [## DVC 介绍:机器学习项目的数据版本控制工具

### 就像 Git 一样，但是有数据！

towardsdatascience.com](https://towardsdatascience.com/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-7cb49c229fe0) 

[https://dvc.org/](https://dvc.org/)