# Python、Anaconda、Google Colab 和虚拟环境入门

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-python-anaconda-google-colab-and-virtual-environments-1ce8fc3286f9?source=collection_archive---------1----------------------->

迟早会有你想学编程的时候。也许你正处于职业转换中，也许你需要一些自动化脚本，或者也许你是一个需要运行[sp leater](https://github.com/deezer/spleeter)的音乐家。也许你被一些流行词汇(人工智能、机器学习、密码、大数据)炒作，或者你只是认为编码很酷。

不管怎样，到时候你决定安装一些 Python 并学习它。酷，欢迎一路顺风！

![](img/6ede00c2725b5cc1f0023a82496e0359.png)

[编程时你唯一能看到的外部风景——由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Fotis Fotopoulos](https://unsplash.com/@ffstop?utm_source=medium&utm_medium=referral) 拍摄]

我就不赘述我探索 Python 的细节了。这里有一个关于如何安装 Python 并开始使用它的简短指南。我将介绍如何:

*   用 **Anaconda** 安装 **Python** (实际上，大部分时间最后只是被称为*Conda*)；
*   使用(Conda) *提示符*；
*   安装新的 *Python 模块*；
*   使用 *Google CoLab* 进行机器学习；
*   用 Python 管理*虚拟环境*。

*编辑:*如果你想在学术和数字人文领域更深入地使用 Python，请查看这本书(这篇文章的其余部分是最初的提纲之一)。

# 让 Python 通过 Anaconda

过去，如果你想开始使用 Python，你必须经常搜索。该语言有两个主要的运行版本:Python 2 和 Python 3。一些语法是不同的(例如，如何在屏幕上打印)，一些主要的库只在一个 Python 版本上运行，等等。

今天你可以省去所有这些谷歌搜索。**去安装 Python 3** (除非你有理由还在用 Python 2，如果你现在开始*就不应该是这种情况)。尽管 Python 版本之争已经结束，但开始使用 Python 仍然有一个违反直觉的问题。开始使用 Python 的最好方法是**而不是**安装 Python。相反，**要安装 Python，你必须谷歌并安装' *Anaconda* '** 。顾名思义，Anaconda 与 Python 有亲缘关系。使用 Anaconda，您将*安装 Python 和其他东西*:*

*   *Python 必须提供给你的大多数科学模块(numpy，pandas，matplotlib…)；*
*   ***IPython**；*
*   **Jupyter 笔记本*支持；*
*   *一个专用的*命令外壳*:Anaconda 提示**；***
*   **Spyder*:Python 的集成开发环境(IDE)；想象一个立体的文本编辑器。*

*即使您对基于 Anaconda 的一站式解决方案不感兴趣，您的 Python 体验仍然需要您使用命令 shell(这对 windows 用户来说有点神秘)。你可能需要使用一些开发工具(更多关于比较 Python IDEs 的主题，请看[这里](https://www.softwaretestinghelp.com/python-ide-code-editors/))。*

*顺便说一句，驱使您使用 Python 的一个主要原因可能是 Anaconda 提供了面向科学的模块作为附带电池。*

# *安装 Anaconda*

*安装 Anaconda 就像去 https://www.anaconda.com/distribution/下载相关的发行版一样简单。如果你像我一样来自 Windows，你需要弄清楚你的系统是 32 位还是 64 位。*

*右击**这台电脑**，选择**属性**，你就会找到答案。(如果这种快速修复不起作用，更多关于确定您运行的是 32 位还是 64 位操作系统的问题，对于所有主要平台，请参见此处的)。在安装 Anaconda 时，系统会询问您安装位置，是否希望它成为运行 Python 文件的默认程序，以及如何与路径变量交互。如果有些事情听起来令人困惑，最好的方法是阅读[安装文档](https://docs.anaconda.com/anaconda/install/)(养成阅读文档的习惯)。关于窗口和路径变量的加分，请参见[这里的](https://en.wikipedia.org/wiki/PATH_(variable))。*

*好了，现在你应该已经安装了 Anaconda 和 Python 了！*

# *了解 Shell 终端:Anaconda 提示符*

*如果你是 Windows 用户，很可能“终端”、“Shell”或“BASH”听起来都不熟悉。在 Windows 10 上有一个电源外壳。搜索并打开它。*

*感觉像是 90 年代或早期的电脑，不是吗？如果你还记得 MS-DOS 时代，那就是了。“提示”是命令行界面的另一种方式。这是你*光盘*要换的目录，你 *dir* 要知道目录里面有什么之类的东西。*

**“这和编程有什么关系？”，*你理所当然地问。如果你想更新你的 Python 版本，安装不同的模块和库等。事实证明，命令行是实现这一点的有效方法。你可以使用 **pip** 给你的 Python 安装更多的包，也就是 pip 安装包(程序员似乎喜欢递归首字母缩写)。*

*如果这听起来令人困惑，有两个好消息:*

1.  *有一个很好的教程叫做[学会足够危险的命令行](https://www.learnenough.com/command-line-tutorial/basics)。这将达到目的，并且当你使用 Git 和 GitHub(这是与世界合作和分享你的工作的伟大工具)时，也会让你感觉很舒服；*
2.  *Anaconda 会让我们变得更容易。Anaconda 自带命令行界面，名为“ **Anaconda Prompt** ”。只需搜索并启动它。你有一个命令行工具。键入' **conda** '以及其他内容，您就可以运行命令并与您的 Anaconda 版本进行交互。*

# *基本 Anaconda 提示符命令*

*Anaconda 提示符命令非常简单:*

*   ***‘conda install’**用于安装包，例如’**conda install pdf miner’**(有时需要添加特定的标志，或者从不同的渠道下载您的包。只要谷歌一下‘康达安装[包]’就能找到详细的说明，比如[https://anaconda.org/conda-forge/pdfminer](https://anaconda.org/conda-forge/pdfminer))；*
*   ***【康达更新】**是您更新整个系统的选择*
*   ***【jupyter notebook】**在你所在的文件夹下创建一个 Jupyter Notebook(更多 Jupyter Notebook 见后文)。*

*要配置虚拟环境，请参见后面的。*

*![](img/b369d4f58a62f4dc1aad48e294bfe0fc.png)*

*[妈妈对 Python 编程的担心是对的…照片由 [Maxx Rush](https://unsplash.com/@maxx07?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄]*

# *通过 Anaconda 运行 Python*

*要通过 Anaconda 运行 Python，您至少有两种选择:*

1.  *在您的系统搜索框中输入' **spyder'** 。然后您将运行 Spyder 应用程序。Spyder 是你将用来编写和运行你的程序的 IDE(集成开发环境)(详见下文)；*
2.  *运行 Python 的另一种方法是打开 Anaconda 提示符并键入' **spyder** '。这将打开 IDE。*

# *Spyder 的优势*

*Spyder 提供了很多好处。我发现最有用的一个是**你可以在不同的单元格中划分你的工作空间**。每个单元格都允许您运行 Python 代码。*

*运行更多 Python 代码单元的重要性在于，你可以有一个主单元，大部分程序都在这个主单元中运行，你可以根据需要试验任意多的单元。如果你需要在主程序中添加一个新的特性，你可以在单元中测试它，然后包含它。使用默认的 Python IDE(称为 IDLE，来自 *Monty Python* show ),你要么打开更多的窗口，要么在运行程序时注释掉不想执行的内容。*

*另外， *Spyder 会在你输入*命令时支持你。按“TAB”键，它会给你一些建议，告诉你如何处理某些对象。这也**帮助你学习**。Spyder 知道你的对象是一个字符串，会告诉你可用的方法。*

*当您开始编程时，可能会出现各种语法错误，这有助于您正确缩进代码。Spyder 检查你的语法，它还可以检查一些格式规则和样式(最著名的是 pep8)。*

# *IPython*

*IPython 是一个*交互式 shell* 来运行 Python 命令。与您在 Spyder 或 Python 编辑器中编写的 Python 代码相反，如果您在 Python shell 中按下“enter ”,就会立即执行。不需要运行它。*

*在 shell 中工作可以让您有更快的响应。你输入命令，看看它们是否如你所愿。在 shell 中构建原型速度很快，并且在运行程序后留在 shell 中查看发生了什么也很有用。IPython 非常适合这一点以及更多。*

*事实上，IPython 中的‘I’代表*交互*。交互性是通过命令的制表符结束(如在 Spyder 中)和让您接近代码来获得的。如果你不知道某个对象的类型，你只需要问 shell，它就会告诉你。只写'**类型(对象)'**。*

*进一步你可以用一个'**？如果你不记得或者你只是好奇，想弄清楚各种函数是做什么的。通过这种方式，你的原型代码*和*与你正在使用的模块的文档进行交互。***

*除此之外，IPython 为您提供了“神奇的方法”,您可以使用这些方法来测量运行代码需要多长时间等等。*

*[这里的](https://ipython.readthedocs.io/en/stable/interactive/)是 IPython 入门教程。*

# *Jupyter 笔记本*

*Jupyter 笔记本是代码和文本的良好交互。您可以在单元或块中运行代码，就像在 Spyder 中一样。尽管如此，在 Jupyter 笔记本中，一个单元格也可以是一个文本块(写于 *Markdown* )。*

*![](img/0fdcd7b7d2113ef75920b36f6bd27656.png)*

*【不是 Jupyter 笔记本。多米尼克·布吕格在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片*

*这允许编写教程，在其中讨论问题，然后展示代码。然后，代码可以在浏览器中运行。Jupyter 笔记本非常适合便携性。不过，你需要在你的机器上*安装依赖项和模块。(有了 Google CoLab，见下文，这不再是问题。)**

*总之，用 Anaconda 打开 Jupyter 笔记本所要做的就是:*

*   *打开**蟒蛇提示**；*
*   *转到您想要创建笔记本的文件夹(即，如果需要，使用**‘CD’**更改目录，使用**‘mkdir’**创建新目录)*
*   *类型**‘jupyter 笔记本’**。*

*笔记本将被创建，您的浏览器将在笔记本中打开。(要关闭笔记本，回到 *conda 外壳*并按下**‘CTRL+C’**。)*

*[这里](https://pythonforundergradengineers.com/opening-a-jupyter-notebook-on-windows.html)是一个关于在 Windows 上用 Anaconda 运行 Jupyter 笔记本的教程，有一步一步的说明和图片。(哦，笔记本有没有看起来那么超级好，还有争议。谷歌“笔记本怀疑论者”出局。)*

# *虚拟环境*

*当你编程时，最好把你正在开发的东西放在隔离的隔间里。你不希望系统更新破坏你的程序，也不希望你下载的新模块与你现有的模块发生冲突。*

*考虑到这一点，把不同的程序放在不同的地方，使它们互不影响，不是很好吗？答案是**建造虚拟环境**。*

*![](img/16a08215ec3cea616b5d7620d3bf8166.png)*

*[虚拟环境变得真实…照片由 [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄]*

*为此，您需要:*

1.  *调用**蟒蛇提示**；*
2.  *键入**' conda create–name[环境名]'** (可以指定想要什么包，什么版本，例如*conda create-n myenv python = 3.6 scipy = 0 . 15 . 0 howdoi，arcade*)；*
3.  *要激活环境，请键入:**' conda activate[环境名称]'**；*
4.  *要停用环境，请键入:**' conda deactivate[环境名称]'**；*
5.  *要有一个所有你创建的环境类型的列表:**‘conda info–envs’**或**‘conda env list’**；*
6.  *要删除已创建的 env，请键入**' conda remove–name[环境名称]–all '**。*

*更多信息，文档再来:[https://docs . conda . io/projects/conda/en/latest/user-guide/tasks/manage-environments . html](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)*

# *蟒蛇和斯派德:激活环境*

*好了，现在我们已经安装了 Python 和 Anaconda，并且熟悉了 Spyder。我们还设法创建了一个虚拟的开发环境。我们需要做的就是在我们的虚拟环境中运行 Spyder。*

*这需要一些进一步的技巧。首先，[读一下这个](https://github.com/spyder-ide/spyder/wiki/Working-with-packages-and-environments-in-Spyder)(特别是在“模块化方法”的标题下)。*

*总而言之，这就是我的工作方式:*

1.  *激活您想要工作的环境(例如使用**激活 Windows 上的 myenv**)。假设您将 arcade 库安装到一个名为 arcade 的环境中。激活它的类型:**‘康达激活街机’**；*
2.  *在那里安装 *spyder-kernels 包*，命令:**' conda install spyder-kernels = 0。***；*
3.  *一旦所有东西都启动并运行了(即创建了 env 并安装了所有需要的东西)。**在环境激活的情况下从 conda 提示符运行 spyder**。为此，键入“spyder3”或“spyder3.exe ”,您将在您的环境中加载依赖项。*

# *网络上的 Google Colab 或 Python*

*开始使用 Python 的最后一个选择是 **Google Colab** 。如果你想试用 Python，又不想搞砸安装，那就去 https://colab.research.google.com/试试 Google Colab。*

*以下是他们对该平台的解释:*

> **什么是协同实验室？Colaboratory，简称“Colab”，允许你在浏览器中编写和执行 Python，零配置要求免费访问 GPU 轻松共享。**

*很好，不是吗？*

*基本上你会有 Jupyter 笔记本的架构，你可以在谷歌的机器上运行。笔记本允许你混合 Markdown 语言(即一种超快速的文字排版方式。想想更快更容易的 HTML 或第一个互联网论坛的语言)和 Python 细胞。*

*如果这看起来还不够令人兴奋，该平台有许多关于数据分析和机器学习的内置教程，如下所示:*

*   ***熊猫简介**:[https://colab . research . Google . com/notebooks/MLCC/intro _ to _ pandas . ipynb](https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb)*
*   ***tensorflow** 编程概念:[https://colab . research . Google . com/notebooks/MLCC/tensor flow _ programming _ concepts . ipynb](https://colab.research.google.com/notebooks/mlcc/tensorflow_programming_concepts.ipynb)*
*   ***图表化**数据:[https://colab.research.google.com/notebooks/charts.ipynb](https://colab.research.google.com/notebooks/charts.ipynb)*

*这有趣吗？请随意在 Linkedin[上联系或在 Substack](https://www.linkedin.com/in/guglielmofeis/) 上加入更广泛的对话。*

*这项工作作为 **CAS 奖学金**的一部分在 *CAS-SEE Rijeka* 进行。点击此处了解更多关于奖学金的信息。*