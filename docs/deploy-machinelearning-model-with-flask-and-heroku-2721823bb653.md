# 用 Flask 和 Heroku 部署机器学习模型

> 原文：<https://medium.com/analytics-vidhya/deploy-machinelearning-model-with-flask-and-heroku-2721823bb653?source=collection_archive---------4----------------------->

建立一个机器学习项目是一回事，但最终重要的是你如何向世界展示你的项目。在 GitHub 上记录你的整个项目是另一回事，但将你的深度学习模型部署为 web 应用程序是完全不同的游戏。

为了让机器学习工程师在工作中取得成功，他们需要构建其他团队可以使用的服务或人们可以直接使用的产品。本质上，目标是提供一个模型作为服务，为此有一个叫做 API 的概念。API 是计算机系统通过互联网协议相互通信的方式。它们充当代理，将信息从用户传到服务器，然后再从服务器传到用户，并返回答案。Flask 提供了这种能力。Flask 将充当模型和 HTML 文件之间的 API。在本文中，我们将关注模型的部署而不是构建。

首先，你需要在本地机器上安装 flask。这可以通过运行命令轻松完成

```
**pip install flask
or
conda install flask**
```

安装完成后，让我们构建一个简单的 Flask 应用程序。

hello-world.py 成功运行

查看构建 Flask 应用程序所需的重要代码行。

```
app=Flask(__name__)
```

这里，我们将 Flask 构造函数赋给一个变量 app，我们需要它来运行所有的进程。您还可以给出静态文件的位置。

来到第二行，

```
@app.route('/',methods=['GET','POST'])
```

**app.route()** 是 Python 中的一个装饰器。在 Flask 中，当你进入一个特定的页面时，每个功能都会被触发。在这里，这个 URL 上的所有流量都将调用我的 **main()** 函数。

您刚刚在 Flask 中完成了第一个应用程序。这不是很简单吗？使用 Flask，我们现在可以包装我们的深度学习模型或机器学习模型，并将它们作为 Web API 提供服务。

让我们继续制作我们的机器学习模型

# 数据清理

你不能直接从原始文本数据去拟合一个机器学习模型。在处理文本数据时，数据清理起着至关重要的作用。当你清理你的数据，它减少了噪音和类似的词，这将减少计算矩阵，从而减少计算时间。

## 停用词的删除

停用词是不会给模型增加价值的常用词，因此您希望删除这些词，这将有助于减少计算矩阵。一些常见的停用词有——我、我们、因为、的等等

即使当你去你的数据集，发现有些词是重复的，你也可以在你的停用词列表中添加这些自定义词。

## 特殊字符的删除

您不希望数据集中出现连字符和点号，因此如果您删除任何您认为不相关的特殊字符，效果会更好。在这里，我删除了所有非字母的字符。

## 词汇匹配/词干提取

这两种技巧都是用来帮助我们达到句子的词根形式。它用于规范化文本。如果你的文本包含**玩**和**玩**，那么我们的模型会认为这是两个不同的单词，但我们知道它们具有相同的重要性，可以简化为一个。然后，就到了使用词干分析还是使用引理分析的问题。两者之间的主要区别是词干化移除了单词的后缀部分，而词汇化回到了词根。

# 向量化文本

我们需要转换我们的数据，以便我们的模型可以理解向量，为此，我们使用 Tfidf 矢量器，这是一种嵌入技术，它考虑了文档中每个单词的重要性。更多关于 TF-IDF 的信息，可以参考[维基百科](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)。

因此，我们已经转换了文本数据，以便我们的机器能够理解。我们使用 tf-idf 将每个单词转换成一个向量。在本教程中，我使用朴素贝叶斯来训练我们的模型。

```
vectorizer = TfidfVectorizer()X = vectorizer.fit_transform(df['Processed'])X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2019)model = MultinomialNB()model.fit(X_train,y_train)model.score(X_test,y_test)
```

最后，我清理了我的模型和矢量器。这是非常重要的，你酸洗你的矢量，因为如果你不这样做，它不会给你想要的结果。

我们现在已经完成了机器学习模型的构建。到目前为止，您会发现很多文章，但只有几篇是关于如何部署它的，所以让我们进入文章的下一部分。

我们已经知道如何创建 Flask 应用程序，让我们构建一个应用程序，它将消息作为输入，并返回它是垃圾邮件还是非垃圾邮件。

我们的模型已经准备好部署了，现在我们必须创建一个 API，让我们开始吧！！

当你创建 API 时，有几件事你必须小心-

1.  从用户处获取请求数据
2.  根据你的需求进行预处理，让你的模型能够理解。
3.  返回响应。如果要返回 JSON 对象，请使用 jsonify。作为标准，主体以 JSON 格式发送。这里我只是返回纯文本，但建议使用 JSON。

首先，我们加载保存的模型和矢量器，我们将在后面的代码中使用它们。正如你看到的 **app.route('/')** 将路由到我的主函数，它将返回我的**main.html**页面，这是我的主页。如果用户点击提交按钮，它将路由到我的/predict，它采用两种方法，即 GET 和 POST，这将触发我的预测功能。我从服务器请求数据，并使用我保存的矢量器将我的文本数据转换成矢量，以便模型可以理解并预测输出。

瞧啊。！您刚刚创建了一个 API，它将接受文本输入并将其分类为垃圾邮件或垃圾邮件。

在这里，它是在本地运行的，但如果我想让全世界都看到我的工作呢？我们如何做到这一点？市场上有很多平台可以部署您的代码。Google App Engine，Heroku，Firebase，AWS EC2 等等。在本教程中，我们将使用 Heroku 来部署我们的代码。与其他产品相比，它的价格非常便宜(大多数时候免费的 dyno 计划就足够了)。

## 文件夹结构

构建代码和文件的结构非常重要。以便更容易导航

```
spam.csv
app.py
model.py
requirements.txt
Procfile
templates/
        main.html
runtime
```

您可以为每个页面添加任意数量的模板，并在新文件夹中添加相同的 CSS 文件。

# 赫罗库

[**Heroku**](https://www.heroku.com/) 是平台即服务(PaaS)，使开发者能够完全在云上构建、运行和操作应用，而不是在你的机器上本地进行。在这个项目中，我们将使用 [heroku git](https://devcenter.heroku.com/articles/git) 进行部署。还有其他方法来部署。

为了使用 heroku git 部署我们的模型，我们需要安装 git 和 heroku CLI。可以参考这些链接安装 [**Git**](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 和**H**[**eroku C**](https://devcenter.heroku.com/articles/heroku-cli)**李**。

在您部署代码之前，您需要在 [*Heroku*](http://www.heroku.com) *上创建一个帐户。*

```
heroku login
```

**创建 Heroku 应用**

要部署 Git 项目，您需要创建一个 Heroku 应用程序。

```
**heroku apps:create unique-name-of-your-app**
```

**需求文件**

这是程序的第一个入口点。它将安装运行代码所需的所有依赖项。requirements.txt 将告诉 heroku 这个项目将需要所有这些库来成功运行应用程序。

**过程文件**

Heroku 要求 Procfile 存在于您的应用程序根目录中。它将告诉 Heroku 如何运行应用程序。确保它是一个没有扩展名的简单文件。Procfile.txt 无效。冒号左边的部分是流程类型，右边的部分是启动该流程所运行的命令。在这种情况下，我们可以判断应该在哪个端口上部署代码，并且您可以启动和停止这些进程。

Procfile

这个文件告诉 heroku 我们想要使用带有命令 gunicorn 和应用程序名称的 web 进程。

**部署到 Heroku**

确保 Procfile 和 requirement.txt 文件存在于您的应用程序根目录中。

```
$ git init
$ heroku git:remote -a unique-name-of-your-app
```

现在，您可以提交 repo 中的文件，并将其推送到主分支

```
$ git add .
$ git commit -am "make it better"
$ git push heroku master
```

更多可以参考 Heroku [文档](https://devcenter.heroku.com/articles/git)。

万岁！你刚刚部署了你的第一个应用。这是一个非常基本的 API，但是如果你正在构建一个更复杂的 web 应用程序，你将需要 JavaScript 的知识。

非常感谢你阅读这篇文章。我希望这会使事情比以前清楚得多。请在评论区建议我如何改进这篇文章。

干杯！祝你学习愉快！！你可以在这里找到这个博客[的源代码](https://github.com/sagardubey3/Spam-vs-Ham-Deployment-using-Heroku)。

# **来源:**

[](https://flask.palletsprojects.com/en/1.1.x/) [## 欢迎使用 Flask - Flask 文档(1.1.x)

### 欢迎阅读 Flask 的文档。开始安装，然后了解快速入门概述。有…

flask.palletsprojects.com](https://flask.palletsprojects.com/en/1.1.x/) [](https://devcenter.heroku.com/articles/procfile) [## Procfile

### Heroku 应用程序包括一个 Procfile，它指定了应用程序在启动时执行的命令。你可以用一个…

devcenter.heroku.com](https://devcenter.heroku.com/articles/procfile) 

# **参考文献:**

[](https://medium.freecodecamp.org/a-beginners-guide-to-training-and-deploying-machine-learning-models-using-python-48a313502e5a) [## 使用 Python 训练和部署机器学习模型的初学者指南

### 当我第一次接触机器学习时，我不知道我在读什么。我读过的所有文章都包括…

medium.freecodecamp.org](https://medium.freecodecamp.org/a-beginners-guide-to-training-and-deploying-machine-learning-models-using-python-48a313502e5a) [](https://blog.usejournal.com/a-guide-to-deploying-machine-deep-learning-model-s-in-production-e497fd4b734a) [## 在生产中部署机器/深度学习模型的指南

### 有大量关于深度学习(DL)或机器学习(ML)的文章，涵盖了数据收集等主题…

blog.usejournal.com](https://blog.usejournal.com/a-guide-to-deploying-machine-deep-learning-model-s-in-production-e497fd4b734a) [](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776) [## 用 Python 开发一个 NLP 模型&用 Flask 逐步部署它

### Flask API，文档分类，垃圾邮件过滤器

towardsdatascience.com](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776) [](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72) [## 自然语言处理实践指南(第一部分)——处理和理解文本

### 经过验证和测试的解决 NLP 任务的实践策略

towardsdatascience.com](https://towardsdatascience.com/a-practitioners-guide-to-natural-language-processing-part-i-processing-understanding-text-9f4abfd13e72)