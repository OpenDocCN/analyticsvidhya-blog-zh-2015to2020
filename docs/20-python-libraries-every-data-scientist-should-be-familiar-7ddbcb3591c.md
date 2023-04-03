# 每个数据科学家都应该熟悉的 20 大 Python 库

> 原文：<https://medium.com/analytics-vidhya/20-python-libraries-every-data-scientist-should-be-familiar-7ddbcb3591c?source=collection_archive---------14----------------------->

嗨，伙计们。在这里，我将列出使用最广泛的 20 个 python 库，它们是我工具箱的一部分，也应该是你工具箱的一部分。所以他们在这里:

# 数据挖掘:

1.  如果你参与了网络抓取，那么这是一个你必须拥有的库。它是一个快速的高级 web 爬行和 web 抓取框架，用于爬行网站并从其页面中提取结构化数据。
2.  [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) :这是另一个流行的网页抓取和数据抓取库。如果你想收集一些网站上可用的数据，但不是在适当的 CSV 或 API 中，BeautifulSoup 可以帮助你收集数据，并将其排列成你需要的格式。
3.  [请求](https://requests.readthedocs.io/en/master/):Kenneth reitz 写的最著名的 HTTP 库。这是每个 python 开发者的必备。它旨在减少人们向 URL 添加查询字符串的工作量，并且是在 Apache2 License 2.0 下发布的
4.  [TeradataSQL](https://pypi.org/project/teradatasql/) :如果你需要从 Teradata 中提取数据，使用这个模块首先需要连接数据库，用 SQL 查询提取数据。使用这个库使数据科学平台的工作变得更加容易。

# 数据处理和建模:

1.  [NumPy](https://numpy.org/) : NumPy 代表“数值 Python”，它是科学计算的核心库，包含对大型多维数组和矩阵的支持，以及对这些数组进行操作的大量高级数学函数。
2.  [Pandas](https://pandas.pydata.org/) :这是一个库，用来帮助开发者直观地处理“标签”和“关系”数据。Pandas 允许将数据结构转换为 DataFrame 对象，处理丢失的数据，从 DataFrame 中添加/删除列，输入丢失的文件，用直方图或绘图框绘制数据。
3.  [SymPy](https://www.sympy.org/en/index.html) : SymPy 可以做代数求值、微分、展开、复数等。它包含在一个纯 Python 发行版中。
4.  [sci kit-Learn](https://scikit-learn.org/stable/):sci kit-Learn 是数据建模和模型评估的最佳库。它包含所有监督和非监督的机器学习算法，并且还带有用于集成学习和促进机器学习的明确定义的函数。
5.  [Statsmodels](https://www.statsmodels.org/stable/index.html) :这是一个 python 模块，它为许多不同的统计模型的估计，以及进行统计测试和统计数据探索提供了类和函数。Statsmodels 建立在数字库 NumPy 和 SciPy 之上，集成了 Pandas 进行数据处理。
6.  [Keras](https://keras.io/) :这是一个很棒的构建神经网络和建模的库。Keras 建立在 Theano 和 TensorFlow Python 库的基础上，这些库提供了构建复杂和大规模深度学习模型的附加功能。此外，微软整合了 CNTK(微软认知工具包)作为另一个后端。
7.  [Tensorflow](https://www.tensorflow.org/) : TensorFlow 是一个开源库，用于跨一系列任务(如对象识别、语音识别和许多其他任务)的数据流编程，由谷歌大脑开发。这是完成任务的最佳工具。该库包括各种层助手(tflearn、tf-slim、skflow)，这使得它的功能更加强大。
8.  [XGBoost](https://xgboost.readthedocs.io/en/latest/index.html) :使用这个库实现梯度提升框架下的机器学习算法。XGBoost 具有可移植性、灵活性和高效性。它提供了并行树提升，可以帮助团队解决许多数据科学问题，尤其是当您处理不平衡的问题时。
9.  [Pytorch](https://pytorch.org/) : PyTorch 是一个非常适合想要轻松执行深度学习任务的数据科学家的框架。该工具允许使用 GPU 加速执行张量计算。它也用于其他任务，例如，创建动态计算图形和自动计算梯度。
10.  [NLTK](https://www.nltk.org/) : NLTK 被认为是分析人类语言和行为的最佳 Python 包。NLTK 库提供易于使用的界面，包含 50 多个语料库和词汇资源，有助于描述人类交互和构建基于人工智能的系统，如推荐引擎。
11.  spaCy :这是一个免费的开源 Python 库，用于实现高级自然语言处理(NLP)技术。与广泛用于教学和研究的 NLTK 不同，spaCy 专注于提供用于生产用途的软件。它具有卷积神经网络模型，用于词性标记、依存解析和围绕训练和更新模型的命名实体识别，并构建自定义处理管道。

# 数据可视化:

1.  [Matplotlib](https://matplotlib.org/) :如果你需要绘图，Matlotlib 是一个选项。它提供了一个灵活的绘图和可视化库，功能强大。这是一个标准的数据科学库，有助于生成数据可视化，如二维图表和图形(直方图、散点图、非笛卡尔坐标图)。
2.  [seaborn](https://seaborn.pydata.org/) : Seaborn 基于 Matplotlib，是一个有用的 Python 机器学习工具，用于可视化统计模型——热图和其他类型的可视化，总结数据并描述整体分布。Seaborn 还内置了面向数据集的 API，用于研究多个变量之间的关系。
3.  pydot :这个库有助于生成有向和无向的图形。它充当 Graphviz(用纯 Python 编写)的接口。在这个库的帮助下，你可以很容易地显示图形的结构。
4.  Bokeh:Python 中交互性最强的库之一，Bokeh 可用于构建 web 浏览器的描述性图形表示。它可以轻松地处理海量数据集，并构建多功能图表，帮助执行广泛的 EDA。散景提供了定义最完善的功能来构建交互式绘图、仪表盘和数据应用程序。
5.  Plotly 是最著名的图形 Python 库之一。它为理解目标变量和预测变量之间的依赖关系提供了交互式图形。使用 Plotly 的 Python API，您可以创建公共/私有仪表板，其中包含图表、图形、文本和网络图像。