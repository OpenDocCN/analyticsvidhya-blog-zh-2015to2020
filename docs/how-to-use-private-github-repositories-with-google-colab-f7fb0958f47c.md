# 如何将代码更改从 google colab 项目推送到私有的 github 存储库？

> 原文：<https://medium.com/analytics-vidhya/how-to-use-private-github-repositories-with-google-colab-f7fb0958f47c?source=collection_archive---------2----------------------->

![](img/21c17eaf696a7b5706f4550fce607b4f.png)

如果你是一个 AI/ML 爱好者，却无法使用 GPU 来训练你的模型，你应该已经知道 google colab 是一个救命稻草。

我个人使用 google colab 进行我的许多机器学习模型，因此我的许多项目都托管在 google drive 上，但是我也喜欢使用 github 作为我所有项目的版本控制系统。

我见过有人在 google colab 上开发他们的机器学习模型，然后，当他们想要将他们的项目上传到 github 的私有存储库时，他们会下载他们的 jupyter 笔记本并重新上传到 github，这既耗时又低效。在相当长的一段时间里，这是唯一的方法，但是 google colab 现在已经扩展了对私有 git 回购的支持。因此，在本指南中，我想我应该解释一下如何做到这一点。

1.  在 google colab 上创建新笔记本。暂且称之为“project0”。“project0.ipynb”将是包含我们实际项目工作的笔记本。
2.  在 google drive 中创建一个新文件夹，并将新创建的笔记本放入其中。我把我的文件夹命名为“GitTutorial”。

![](img/d68bde9fbabc55f4398d53659a1813d8.png)

2.现在创建另一个笔记本，并将其放在同一文件夹中。这个笔记本将包含允许我们将工作上传到 github 的命令。姑且命名为“git.ipynb”。

![](img/08cc78c02b97f6d49cb7a8dddc719853.png)

3.现在，你可以在“project0.ipynb”中处理你的项目，每当我们需要将我们的工作推送到 github 时，我们将使用“git.ipynb”。

4.我们需要在我们的 colab 笔记本上安装 google drive。这很容易做到，在一个单元格中编写下面的代码，运行它，复制授权代码，粘贴到提供的空间，你就完成了。

```
from google.colab import drivedrive.mount('/content/gdrive')
```

![](img/c2744929baf491d043708b21b23dea5f.png)

5.现在我们需要导航到我们的 google drive 项目文件夹。

```
%cd gdrive/My Drive/%cd GitTutorial
```

6.最后，我们可以开始实际的 git 工作了。我们需要做的第一件事是初始化一个本地 git 存储库(即在我们的 google drive 项目文件夹中)

```
!git init
```

7.现在创建一个私有的 git 存储库。我将我的库命名为“gitTutorial ”,然后我们需要将这个库作为远程库添加到我们的 git 项目中。

```
!git remote add origin [https://{github](https://{github)_username}:{github_password}@github.com/{github_username}/{github_repo_name}.git!git remote -v
```

用您的详细信息替换占位符。例如，如果我的 github 用户名是 saifullah73，我的密码是 12345，我的库名是 gitTutorial，那么上面的命令对我来说就是"！git 远程添加原点*https://saifullah 73:12345 @ github . com/saifullah 73/git tutorial . git "*

8.我们还需要在对代码进行任何修改之前告诉 git 我们是谁。为此运行以下代码。

```
!git config --global user.email '{user_email}'!git config --global user.name '{user_name}'
```

*用您自己的详细信息替换{您的电子邮件}和{用户名}，例如。*

*！git config-global user . email ' saifullakhaliq @ Gmail . com '*

*！git 配置—全局用户名' saifullah'*

8.我们现在已经准备好对项目进行第一次提交了。

```
!git add .!git commit -m 'First commit'
```

9.推送至 github。

```
!git push origin master
```

*“master”是我们要将代码推送到的远程分支的名称，如果该分支不存在，它将被自动创建。*

10 .瞧啊。我们已经成功地将我们的 colab 项目与 private git repo 集成在一起。

![](img/b81e2daf0568d4d4d277f502f1035312.png)

一些需要记住的事情。

每当您需要在稍后阶段推送时，您将需要从您的远程存储库中提取。为此，只需运行下面的代码。一定要在拉之前提交代码。

```
!git pull origin master
```

在此之后，您可以提交和推送您的更改。

我一直保持这个教程非常简单，只涵盖了基础知识，这绝不是一个先进的教程，更多的细节你可以查看以下几个链接。

[](https://www.tutorialspoint.com/git/index.htm) [## Git 教程

### Git 是一个强调速度的分布式版本控制和源代码管理系统。Git 最初是…

www.tutorialspoint.com](https://www.tutorialspoint.com/git/index.htm) [](https://try.github.io/) [## 学习 Git 的资源

### Git，GitHub，DVCS，天啊！学习 Git 的所有行话和基础知识。

try.github.io](https://try.github.io/) 

你可以在 https://github.com/saifullah73/GitTutorial 找到本教程的代码

我的 linkedIn

https://www.linkedin.com/in/saifullah73/