# 开始使用 Plotly @ Jupyter Lab

> 原文：<https://medium.com/analytics-vidhya/get-started-with-plotly-jupyter-lab-b280ddcc6008?source=collection_archive---------12----------------------->

让魔法开始吧！

什么是**剧情**？这是一个交互式 Python 图形库。您可以使用 Plotly 制作许多美丽而迷人的图表，如折线图、散点图、箱线图、直方图和热图。

下面是我用 Plotly 做的一个地理热图的例子。

![](img/9156fcf4522009772b8a0027b4310f16.png)

您可以悬停在每个州查看数据

# 库安装

让我们从库安装开始，您必须在 Jupyter 实验室中运行以下代码:

```
#pip install -U plotly
```

你可以检查你的 Plotly 版本:

```
from plotly import __version__
print(__version__)
```

# 扩展安装

在库安装之后，您需要安装扩展`@jupyterlab/plotly-extension`，以便在 JupyterLab 中使用`iplot`显示绘图。或者在运行 Plotly 代码后，你会得到一个巨大的空白单元格输出，并且绞尽脑汁想知道哪里出错了。

为了在 JupyterLab 中使用，使用 pip 安装`jupyterlab`和`ipywidgets`包...

```
pip install jupyterlab==1.2 "ipywidgets==7.5"
```

如果没有的话，安装`[node](https://nodejs.org/)`。

运行以下命令安装所需的 JupyterLab 扩展:

```
# Avoid "JavaScript heap out of memory" errors during extension installation
# (OS X/Linux)
export NODE_OPTIONS=--max-old-space-size=4096
# (Windows)
set NODE_OPTIONS=--max-old-space-size=4096# Jupyter widgets extension
jupyter labextension install @jupyter-widgets/jupyterlab-manager@1.1 --no-build# FigureWidget support
jupyter labextension install plotlywidget@1.4.0 --no-build# and jupyterlab renderer support
jupyter labextension install jupyterlab-plotly@1.4.0 --no-build# Build extensions (must be done to activate extensions since --no-build is used above)
jupyter lab build# Unset NODE_OPTIONS environment variable
# (OS X/Linux)
unset NODE_OPTIONS
# (Windows)
set NODE_OPTIONS=
```

在我的例子中，我从未成功运行过`jupyter lab build`，但是当我打开 Jupyter Lab 时，它会提示我允许构建并在构建完成后重新加载页面。

你可能会看到[https://github . com/plotly/plotly . py # jupyterlab-support-python-35 1.0k](https://github.com/plotly/plotly.py#jupyterlab-support-python-35)完整的安装说明。如果不起作用，您可以尝试以下步骤之一:

*   清理 Jupyter 实验室建筑

```
jupyter lab clean
```

*   卸载 Jupyter Lab 并重新安装

```
# Uninstall
pip uninstall jupyter# Type 'y' to confirm installation# Install it again to start fresh
pip install jupyter
```

希望这有帮助！