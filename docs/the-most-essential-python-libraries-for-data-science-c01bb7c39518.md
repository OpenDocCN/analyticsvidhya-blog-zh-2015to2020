# 数据科学最基本的 Python 库

> 原文：<https://medium.com/analytics-vidhya/the-most-essential-python-libraries-for-data-science-c01bb7c39518?source=collection_archive---------4----------------------->

![](img/020f7f4420da6d89e2cf23f7da26c679.png)

照片由 [**麦克斯韦**](https://unsplash.com/@maxcodes?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上[下**上**下](https://unsplash.com/s/photos/python?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

随着互联网上可找到的数据量的增加，Python 最近也出现了巨大的使用热潮[](https://www.youtube.com/watch?v=YqxeLodyyqA)**(从 2010 年的 9%到 2020 年的 25%)。这反过来导致数据科学家、机器学习工程师和数据分析师的工作机会[](https://www.forbes.com/sites/louiscolumbus/2017/05/13/ibm-predicts-demand-for-data-scientists-will-soar-28-by-2020/?sh=57ee81387e3b)**增加(平均工资也相当不错)。****

****基本上，Python 已经在数据科学领域流行了一段时间，因此是技术行业的一项热门技能。****

****要在 Python 中执行任何基于数据科学的任务，需要导入 [**包**](https://www.python-course.eu/python3_packages.php#:~:text=A%20package%20is%20basically%20a,several%20modules%20into%20a%20Package.) **，**也称为库。这些软件包可以帮助您高效地实现目标，让您的工作变得更加轻松。包裹基本上是蝙蝠侠的知更鸟，除了在这种情况下有太多知更鸟，一些好的，一些伟大的，一些…他们完成了任务。****

****在这个故事中，我试图成为你的阿尔弗雷德，并根据我的经验和探索，帮助你筛选和找到对数据科学至关重要的知更鸟，并帮助你实现你的目标。****

****![](img/4e27ea47ae8b86d3fa6587e800bd6dc0.png)****

# ******数学和科学计算库******

# ******1。NumPy******

****NumPy 是科学应用程序中最常用的库之一，用于高效地操作大型多维数组的任意记录，而不会牺牲太多的速度。它提供了处理数组对象的工具。****

****它可以用来做，****

*   ****算术运算****
*   ****处理复数****
*   ****指数运算****
*   ****三角运算****

****以及更多的数学过程。它在诸如切片、索引、分割、广播等操作过程中也很方便。****

****NumPy 还可以用来读写文件，在机器学习的情况下，它也是一个重要的数据预处理工具。****

****应该注意的是，NumPy 主要处理同构多维数组。****

****最新版本: [**1.19.4**](https://pypi.org/project/numpy/) 。****

****在这里 阅读更多关于这个库 [**及其功能**](https://numpy.org/) **[**在这里**](https://numpy.org/doc/stable/reference/routines.math.html) 。******

# **2.SciPy**

**SciPy 非常像 NumPy，是一个用于数学、科学和工程目的的开源实现。SciPy 依赖于 NumPy，因为它在 NumPy 数组上工作并大量使用它。**

**SciPy 包是由像 NumPy、Matplotlib、Pandas、SymPy、IPython 和 nose 这样的库组成的 SciPy 栈的一部分。**

**当涉及到计算时，SciPy 具有帮助执行数学例程的模块，**

*   **线性代数**
*   **微分方程**
*   **综合**
*   **信号处理**
*   **统计分析，以及**
*   **插入文字**

**以有效的方式。**

**最新版本: [**1.5.3**](https://pypi.org/project/scipy/) 。**

**更多了解本库 [**此处**](https://www.scipy.org/) 及其功能 [**此处**](https://docs.scipy.org/doc/scipy/reference/) 。**

# **3.统计模型**

**Statsmodels 为数学计算提供了对 SciPy 的补充，但它被广泛使用，是统计计算、描述性统计和统计模型的估计和推断的最佳选择。**

**Statsmodels 使得在 Python 本身上执行统计操作变得非常容易和方便。当涉及到统计学时，R 是最好和最容易使用的，这一点已经被广泛接受。Statsmodels 试图在 Python 中提供类似的易用性。**

**统计模型可用于**

*   **线性回归模型**
*   **多元计算**
*   **时间序列分析**
*   **假设检验**
*   **混合线性模型、广义线性模型和贝叶斯模型**

**最新版本: [**0.12.1**](https://pypi.org/project/statsmodels/) **。****

**在这里 阅读更多关于这个库 [**及其功能**](https://www.statsmodels.org/stable/index.html) **[**在这里**](https://www.statsmodels.org/stable/user-guide.html) 。****

# **4.熊猫**

**是的，它们是以中国可爱的动物命名的，而且它们非常有用。**

**![](img/e18f97f4cc20e0b4a0a6db145f22759a.png)**

**照片由 [**参宿七**](https://unsplash.com/@rigels?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上 [**下**](https://unsplash.com/s/photos/pandas?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)**

**Pandas 被广泛认为是使用 Python 的应用数据科学的最重要的库。**

**你可以用熊猫完成大量的功能。它提供了快速、灵活和富有表现力的数据结构，使得处理大型结构化和时序数据变得非常容易。**

**基本上，数据取自 SQL 表或 CSV 文件，并存储在数据帧中，使用这些数据帧，您可以以有助于获得所需结果的方式轻松操作数据。最接近熊猫数据框架的是 Excel 表格。**

**您可以使用熊猫执行的一些功能有:**

*   **从平面文件(CSV 和分隔文件)、Excel 文件、数据库加载数据**
*   **将数据写入/保存到上述类型的文件和数据库中**
*   **数据集的基于标签的切片、花式索引和子集化**
*   **数据集的合并和连接**
*   **时间序列功能**

**有趣的事实:Pandas 代表 ***Python 数据分析库*** 。**

**最新版本: [**1.1.4**](https://pypi.org/project/pandas/) **。****

**更多了解本库 [**此处**](https://pandas.pydata.org/) 及其功能 [**此处**](https://pandas.pydata.org/docs/user_guide/index.html#user-guide) 。**

# **可视化库**

# **1.Matplotlib**

**当我第一次听说 Matplotlib 时，我认为它听起来像 Matlab，因此会提供类似的功能。嗯，我并不完全正确。Matplotlib 用于绘制图形和制作图表，也是 SciPy 堆栈的一部分。它也可以用来显示图像。**

**Matplotlib 是 Python 的一个库，它提供了一个面向对象的 API，用于将绘图嵌入到应用程序中。使用 Matplotlib，您可以将数据可视化，编写故事，并以一种清晰明确的方式展示您的发现。用于绘制这些图形的代码在语法上类似于 Matlab 中的代码。**

**Matplotlib 还允许你格式化网格、标签、标题、图例和图形的其他组件。**

**它可以用来制作，**

*   **曲线图**
*   **散点图**
*   **面积图**
*   **直方图**
*   **条形图**
*   **饼图**
*   **等高线图**
*   **箱线图**

**还有更多！！！**

**最新版本: [**3.3.2**](https://pypi.org/project/matplotlib/) **。****

**在这里 阅读更多关于这个库 [**及其功能**](https://matplotlib.org/) **[**在这里**](https://matplotlib.org/gallery/index.html) 。****

# **2.海生的**

**Seaborn，简单来说就是 Matplotlib 2.0。它基于 Matplotlib，提供了一个绘制有吸引力的、信息丰富的统计图形的高级界面。**

**![](img/87f60a9ee4c1541553a7c0ce0ae05626.png)**

**照片由 [**爱德华·豪威尔**](https://unsplash.com/@edwardhowellphotography?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上[**Unsplash**](https://unsplash.com/s/photos/graphs?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)**

**所以你可能会问这两者有什么区别…好吧我(实际上是 [**这篇**](https://analyticsindiamag.com/comparing-python-data-visualization-tools-matplotlib-vs-seaborn/#:~:text=Matplotlib%3A%20Matplotlib%20is%20mainly%20deployed,has%20easily%20interesting%20default%20themes.) 文章)已经覆盖了:Matplotlib 主要用于绘制基本图形，如条形图、饼图、线图、散点图等。另一方面，Seaborn 提供了多种复杂的可视化模式，同时使用了较少的语法。**

**除了常见的图形(条形图、散点图、折线图、面积图、饼图等)之外，Seaborn 还可以用于制作，**

*   **联合分布图**
*   **密度图**
*   **因子图**
*   **群体图**
*   **小提琴情节**
*   **棒棒糖图**

**还有更多！！**

**最新版本: [**0.11.0**](https://pypi.org/project/seaborn/) **。****

**在这里 阅读更多关于这个库 [**及其功能**](https://seaborn.pydata.org/) **[**在这里**](https://seaborn.pydata.org/tutorial.html) 。****

# **3.Plotly**

**Plotly 是一个 Python 图形库，可用于制作交互式图形。与其他可视化工具一样，数据被导入，然后被可视化和分析。**

**Plotly 更像是 Matplotlib 和 Seaborn 的“企业”版本，因为它可以集成并用于构建面向 ML 或数据科学的 web 应用程序。**

**Plotly 可用于，**

*   **制作基本图表(散点图、折线图、条形图、饼图、气泡图、甘特图)**
*   **制作统计图表(箱线图、直方图、散点图、误差棒图、格子图、小提琴图)**
*   **制作财务图表(烛台，瀑布，漏斗，OHLC)**
*   **使用地图进行可视化**
*   **支线剧情**
*   **三维图表**
*   **使用各种转换(聚合、分组、过滤)制作图表**
*   **Jupyter Widgets 交互**

**最新版本: [**4.12.0**](https://pypi.org/project/plotly/) **。****

**在这里 阅读更多关于这个库 [**及其功能**](https://plotly.com/) **[**在这里**](https://www.journaldev.com/19692/python-plotly-tutorial) 。****

# **4.散景**

**Bokeh 是一个 Python 库，可用于交互式可视化。**

**它的主要功能之一是可以用来可视化学习算法。对于每个刚刚开始学习机器学习概念的人，我会推荐散景，因为它可以帮助你更好地理解简单的 ML 技术，如 K-means 或 KNN。**

**它的应用是，**

*   **创建基本图表**
*   **快速轻松地制作交互式图表、仪表盘和数据应用程序**
*   **它支持 [**HTML**](https://docs.bokeh.org/en/latest/docs/user_guide/embed.html) ， [**Jupyter 笔记本**](https://docs.bokeh.org/en/latest/docs/user_guide/jupyter.html) 或 [**服务器**](https://docs.bokeh.org/en/latest/docs/user_guide/server.html) 输出**
*   **可视化可以集成到 Django 和 Flask 应用程序中。**

**最新版本: [**2.2.3**](https://pypi.org/project/bokeh/) 。**

**在这里 阅读更多关于这个库 [**及其功能**](https://docs.bokeh.org/en/latest/index.html) **[**在这里**](https://docs.bokeh.org/en/latest/docs/user_guide/plotting.html) 。****

# **机器学习库**

# **1.张量流**

**Tensorflow 是数据爱好者使用的最流行的深度和机器学习框架。它是免费的，是由谷歌大脑团队开发的开源软件库。**

**Tensorflow 使用起来非常简单，可以用来开发和部署机器学习应用。它允许您使用具有许多层的神经网络，这得益于 GPU 集成，它可以帮助您在一台机器上的多个 GPU 上运行估计器模型。**

**Tensorflow 的一些最受欢迎的用途和应用是:**

*   **语音识别系统**
*   **文本摘要**
*   **图像/视频识别和标记**
*   **情感分析**
*   **自动驾驶汽车**
*   **推荐系统。**

**使用 Tensorflow，一些非常酷的应用程序已经被开发出来，下面是其中的一些:**

*   **[**盗图分类器**](https://ai.googleblog.com/2016/03/train-your-own-image-classifier-with.html) 由谷歌**
*   **斯坦福大学 [**用于药物发现的大规模多任务网络**](https://arxiv.org/pdf/1502.02072.pdf)**
*   **谷歌创意实验室 [**Nsynth Super**](https://nsynthsuper.withgoogle.com/)**
*   **[**深度演讲**](https://github.com/mozilla/DeepSpeech) 由 Mozilla**

**最新版本: [**2.3.1**](https://pypi.org/project/tensorflow/) 。**

**阅读更多关于本库 [**这里**](https://www.tensorflow.org/) 及其功能 [**这里**](https://www.tensorflow.org/tutorials) 。**

# **2.克拉斯**

**Keras 是一个高级库，可以从 Tensorflow 导入。Keras 简化了许多任务，将您从编写大量单调的代码中解放出来，但它可能不适合复杂的过程或任务。**

**由于 Keras 运行在 Tensorflow 之上，问题就出现了:这两个库有什么区别？**

**首先，Keras 没有 Tensorflow 那么复杂，简单易用。Keras 也是一个高级 API，作为 Tensorflow 和 Theano 的包装器。如果你只是对创建和执行机器学习模型感兴趣，Keras 适合你，但如果你也想知道更深层次的复杂性和工作方式，那么 Tensorflow。**

**Keras 的一些应用包括:**

*   **图像分类**
*   **特征抽出**
*   **微调和损耗计算**

**最新版本: [**2.4.3**](https://pypi.org/project/Keras/) 。**

**在这里 阅读更多关于这个库 [**及其功能**](https://keras.io/) **[**在这里**](https://keras.io/getting_started/intro_to_keras_for_researchers/) 。****

# **3.sci kit-学习**

**Scipy 是一个免费、开源的 Python 机器学习库，使用 NumPy、SciPy 和 Matplotlib 构建。**

**它提供了各种有监督和无监督的机器学习模型，是 ML 初学者的理想选择。文档非常简单直观，最重要的是，非常简洁。使用很少几行代码，您就可以训练一个模型，然后实现它。**

**它是处理数据的最佳库之一。**

**Scikit-Learn 帮助您:**

*   **分类模型(SVM、最近邻、随机森林、朴素贝叶斯、决策树、监督神经网络等)**
*   **回归模型(SVM、最近邻、朴素贝叶斯、决策树、监督神经网络等)**
*   **使聚集**
*   **降维**
*   **型号选择**
*   **数据预处理**

**最新版本: [**0.23.2**](https://pypi.org/project/scikit-learn/) 。**

**在这里 阅读更多关于这个库 [**及其功能**](https://scikit-learn.org/stable/) **[**在这里**](https://scikit-learn.org/stable/user_guide.html) 。****

# **4.NLTK**

**说到文本或者文本分析，机器学习的一部分就是*自然语言处理*，这恰好是 NLTK 的强项。**

**![](img/f1f985278b7c8ee434c84a82c5c65a2c.png)**

**照片由 [**梁锦松**](https://unsplash.com/@ninjason?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 于 [**上 Unsplash**](https://unsplash.com/s/photos/text?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)**

**NLTK，代表自然语言工具包，是许多库的组合，帮助您分析和处理文本，以便仅使用文本就能得出有意义的结论。**

**必须注意，NLTK 仅用于处理数据。它不提供任何机器学习模型。你可以使用 LSTM 或伯特或深度学习模型，进行后处理，以训练一个以文本为输入得出结果的模型。**

**NLTK 的主要特性和用途是:**

*   **词干化和词汇化**
*   **标记化**
*   **磨尖**
*   **情感分析**
*   **主题建模/文本分类。**

**最新版本: [**3.5**](https://pypi.org/project/nltk/) 。**

**在这里 阅读更多关于这个库 [**及其功能**](https://www.nltk.org/) **[**在这里**](/@ramithaabeyratne/basic-functions-of-nltk-q-a-approach-81d44baf403c) 。****

# **用于抓取数据的库**

# **1.Scrapy**

**Scrapy 是一个“*开源协作框架”*，可以用来从网站中提取数据，即网页抓取。它提供了一个快速而强大的框架来从网页中提取你需要的信息。它被认为是 Python 最好的网络爬虫。**

**Scrapy 最好的一点是它的可扩展性，因为它可以用来从 API 中提取数据，并且可以在不触及核心的情况下插入功能。**

**你可能会问为什么要用 Scrapy，让我来告诉你为什么:**

*   **使得抓取任何网站成为可能**
*   **请求是异步调度和处理的**
*   **可以直接从提供 JSON 数据的网站解码 JSON**
*   **利用扫描网页和收集结构化数据的蜘蛛机器人。**

**必须注意的是，Scrapy 只能在 Python 2.7 及以后的版本中使用。**

**最新版本:[T3 2 . 4 . 0T5。](https://pypi.org/project/Scrapy/)**

**在这里 阅读更多关于这个库 [**及其功能**](https://scrapy.org/) **[**在这里**](https://docs.scrapy.org/en/latest/intro/tutorial.html) 。****

# **2.美丽的声音**

**很美，但绝对不是汤。**

> **那糟糕的一页不是你写的。你只是想从中获取一些数据。美丽的汤是来帮忙的。**

**如果你去他们的 [**网站**](https://www.crummy.com/software/BeautifulSoup/) ，这些会是最开始的几句话。老实说，还有什么好说的。**

**一定要看看他们的 [**名人堂**](https://www.crummy.com/software/BeautifulSoup/#HallOfFame) 页面，看看一些用 BeautifulSoup 制作的高调项目。**

**美丽的声音，**

*   **易于使用和掌握**
*   **语法非常简单，文档非常清晰，信息丰富**
*   **与 Scrapy 相比，它是一个较小的库，因此需要最小的设置和较少的关注。**

**此外，与 crawler 不同，BeautifulSoup 使用 HTML 解析，这基本上意味着您必须提供某种 HTML 地址才能获取信息。**

**最新版本: [**4.9.3**](https://pypi.org/project/beautifulsoup4/) 。**

**在这里 阅读更多关于这个库 [**及其功能**](https://www.crummy.com/software/BeautifulSoup/) **[**在这里**](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 。****

**祝您在数据科学之旅中好运，感谢您的阅读:)**