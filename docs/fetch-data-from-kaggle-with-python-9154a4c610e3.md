# 用 Python 从 Kaggle 获取数据

> 原文：<https://medium.com/analytics-vidhya/fetch-data-from-kaggle-with-python-9154a4c610e3?source=collection_archive---------1----------------------->

## 关于如何用代码连接 Kaggle 的简短有用的信息。

![](img/acf5d36d8d30d43582b45cb4d9f9c855.png)

安德鲁·庞斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

如今，基于世界局势，大多数分析都多少涉及到新冠肺炎研究。或许，每一个对疫情传播和“行为”稍有兴趣的公司都要求数据分析师获取数据并进行分析。由于新冠肺炎是每天都有更新的数据，当你可以在笔记本上有关于代码的**【所有在一个地方】**时，它就方便了。

让我向您展示如何通过 Python 代码与 Kaggle 页面进行交互。

这篇文章很简单，但是很有用。

## 将涵盖的主题:

1.用 API 连接到 Kaggle

2.与竞赛互动

3.与数据集交互

开始编码吧！

# 用 API 连接到 Kaggle

首先，如果你想与 Kaggle 互动，你必须注册并拥有一个帐户。有了帐户，您就可以参加竞赛，查看其他笔记本、编码，还可以下载数据集用于您的分析和其他转换。

在我们探索更多之前，为了访问 Kaggle API，我们需要安装它:

```
!pip install kaggle
```

您也可以在命令提示符下使用以下命令来完成此操作:

```
pip install kaggle
```

现在，我们需要创建 API 凭证——这非常简单。在您的 Kaggle 帐户上，在 API 下，选择“创建新的 API 令牌”，然后 **kaggle.json** 将被下载到您的计算机上。

转到目录—“C:\ Users \<username>”。kaggle\" —并将下载的 JSON 文件粘贴到这里。</username>

使用以下代码，您可以连接到 Kaggle:

```
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
```

你已经准备好探索卡格尔的世界了。

# 与竞赛互动

## 列出正在进行的竞赛:

只需一行代码，您就可以列出所有想要的匹配项，并且可以在搜索时输入非常酷的东西。

参数 **category=' '** 使您能够选择您要寻找的比赛类型，请记住可用选项有:

> 所有'，'特色'，'研究'，'招聘'，'入门'，'大师'，'游乐场'

```
api.competitions_list(category='gettingStarted')
```

如果我们将**类别**设置为‘getting started ’,那么我们会得到以下列表:

```
[digit-recognizer,
 titanic,
 house-prices-advanced-regression-techniques,
 connectx,
 nlp-getting-started,
 facial-keypoints-detection,
 street-view-getting-started-with-julia,
 word2vec-nlp-tutorial,
 data-science-london-scikit-learn,
 just-the-basics-the-after-party,
 just-the-basics-strata-2013]
```

但是，如果我们已经在 Kaggle 网站上搜索了比赛，并且我们想要输入一个特定的比赛，该怎么办呢？

然后，我们使用**搜索**参数来列出名称中包含搜索词的特定竞争或多个竞争:

```
api.competitions_list(search='titanic')
```

让我们看看如果我们输入搜索“NLP”会出现什么:

```
api.competitions_list(search='nlp')
```

输出:

[NLP-入门，
google-quest-challenge，
jigsaw-unintended-bias-in-toxity-class ification，
gender-代词-resolution，
word2vec-nlp-tutorial，
data-science-for-good-city-of-los-Angeles]

我们可以将输出保存到一个列表中，并选择想要的比赛进行探索:

```
nlp_list=api.competitions_list(search='nlp')
```

选择谷歌-探索-挑战:

```
nlp_list[1]
```

## 从所选竞争中获取数据:

为了进一步探索，我将使用泰坦尼克号比赛作为一个例子。

在下载数据之前，我们可以列出所选竞争中可用的所有数据集:

```
api.competition_list_files('titanic')
```

输出:[train.csv，test.csv，gender_submission.csv]

下载与代码竞争的数据集:

```
api.competition_download_files('titanic')
```

你可以看到下载的文件是一个 zip 文件。只需几行代码，我们就可以将其解压缩到所需的目录中:

```
from zipfile import ZipFile
zf = ZipFile('titanic.zip')
zf.extractall('data\') #save files in selected folder
zf.close()
```

在我的例子中，我选择将数据集保存在数据文件夹中。您可以通过键入正确的文件夹名称而不是数据来选择您的文件夹。如果您想要提取笔记本所在的当前文件夹中的数据集，只需擦除目录，如下所示:

```
zf.extractall()
zf.close()
```

最后，我们导出了数据集以供进一步分析。

## 服从竞争

每场比赛的结果都由参赛者提交的文件进行评估。也可以直接从 Python 提交您的产品:

```
api.competition_submit(#name of saved results
'gender_submission.csv','API Submission',#name of competition
'titanic')
```

这就是关于 Kaggle 比赛的全部内容。我希望你喜欢它，但等一下还有更多！

# 与数据集交互

你仍然可以下载和使用数据而不用参加比赛。有时，我们只是想分析一个令人兴奋的数据集，或者获得对特定问题/任务的更多见解。

通过代码进行交互很方便，因为所有的事情都在同一个笔记本**中完成**。不需要在边上下载数据集，然后将它们导入到想要的目录。

说到这里，我们开始吧！

如果要下载指定 Kaggle 数据下的所有文件，请使用:

```
#downloading datasets for COVID-19 data
api.dataset_download_files('imdevskp/corona-virus-report')
```

**括号内的路径**为—用户名(of _ person _ who _ published _ dataset)/姓名(of_the_Kaggle_dataset)。

如果你现在感到困惑，我能理解。Kaggle 数据集可以包含多个数据集，如果我们定义了“唯一”路径，那么所有可用的数据集都将从 Kaggle 数据集下载。

我们还可以**指定**要下载哪些数据集，具体如下:

```
api.dataset_download_file('imdevskp/corona-virus-report','covid_19_clean_complete.csv')
```

添加所需数据集的名称将会阻止代码下载所有可用的数据集，但只有一个**被选中**。

同样，在这两种情况下我们都需要解压缩它，我们按照上面解释的那样做。小小的重复不会伤害任何人，所以我们开始吧:

```
zf = ZipFile('covid_19_clean_complete.csv.zip')
#extracted data is saved in the same directory as notebook
zf.extractall() 
zf.close()
```

现在，有了**熊猫**，你可以轻松地加载数据集并开始使用它们。

```
import pandas as pd
data=pd.read_csv('covid_19_clean_complete.csv')
```

# 结论

你可以看到这并不复杂，但可以节省你很多时间，特别是当每天下载一个更新的数据集。

让我知道你是用这种方式还是仅仅通过 Kaggle 网站下载？

老实说，根据使用所选数据集的情况和频率，我两种方式都用。

我希望你学到了一些东西，并喜欢阅读它。

注意安全。再见。

你可以找到更多关于 https://github.com/Kaggle/kaggle-api 的信息。