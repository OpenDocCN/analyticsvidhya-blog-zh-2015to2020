# 利用多元线性回归和消费价格指数预测模型进行预期寿命预测

> 原文：<https://medium.com/analytics-vidhya/life-expectancy-prediction-using-multiple-linear-regression-and-consumer-price-index-forecasting-174d1498257d?source=collection_archive---------15----------------------->

O *原写于:2020 年 4 月 5 日*

*(最初作为“数据分析的统计”模块的首次连续评估提交)*

博客原文链接:[https://astrum-imber . blogspot . com/2020/07/life-expectation-prediction-using . html](https://astrum-imber.blogspot.com/2020/07/life-expectancy-prediction-using.html)

# 摘要

> 本报告描述了一个基于多元线性回归的模型，用于预测联合国(UN)预期寿命数据集中所列国家公民的预期寿命。该模型使用各种其他联合国数据集来选择因变量(该年的国家总人口、该年的国内生产总值、该年的艾滋病毒死亡率等)。)然后检查执行回归分析所需的所有必要假设。该报告还介绍了一个时间序列模型，该模型利用欧洲联盟(欧盟)数据集储存库提供的数据集，采用 ARIMA 模型预测了 20 国集团国家的消费价格指数。这个模型的数据清理是在 Python 上完成的，而统计分析和模型开发是在 IBM SPSS 和 R Studio 上完成的。与其他指标一起，使用的主要评估指标是 R 平方和 p 值测试。
> 
> **关键词**-预期寿命，消费价格指数，回归，预测，统计分析。

# 一.导言

由于现代医学和医疗设施的进步，人类活得更长、更健康、更充实。患有慢性疾病但仍能轻松完成日常生活功能，现在比以往任何时候都更有可能。虽然医学知识有助于提高全世界人类的平均预期寿命，但数据收集和分析方面的进步使得通过设计相关模型来研究其影响成为可能。硬币的另一面是对自然和经济资源不断增加的压力。这使得有必要分析和衡量这种技术注入对各国经济的影响。其中一个衡量标准是消费价格指数，它将各种家庭用品的价格变化作为一个通货膨胀单位(Bryan 和 Cecchetti，1993 年)(Dougherty 和 Van Order，1982 年)。研究技术对消费者日常生活的影响可以帮助行政部门做出关键决策和政策改变，以改善公民的生活。

本报告提出了一个基于多元线性回归统计技术的模型，该模型分析了可能影响也可能不影响某一年某一国家预期寿命准确预测的不同因素。这份报告还试图通过模拟和拟合多年来的趋势来预测 20 国集团国家的通货膨胀指数(或 CPI)的增长。

# 二。数据源、清理和准备

## A.使用的变量和数据来源

*因变量:*出生时预期寿命(年)

1.  按现价计算的人均国内生产总值
2.  国家总人口
3.  政府卫生总支出占政府总支出的百分比
4.  1 岁儿童卡介苗免疫覆盖率(%)
5.  死亡——艾滋病毒/艾滋病(年龄标准化)(比率)(每 10 万人)
6.  1 岁儿童脊髓灰质炎(Pol3)免疫接种覆盖率(%)
7.  1 岁儿童乙肝疫苗接种率(%)
8.  成人死亡率(每 1000 人中 15 至 60 岁之间死亡的概率)
9.  人类发展指数

数据来源:使用的所有数据集都直接从联合国数据集在线储存库(data.un.org，n.d .)下载。csv 文件格式。

使用的变量:消费者价格指数

数据来源:所使用的数据集直接从欧盟统计局网上资料库(ec.europa.eu，n.d .)下载。csv 文件格式

## B.清洁和准备

因变量和自变量的数据集以 csv 格式存储，并导入 Jupyter 笔记本进行清理和转换。Python 包，即 Pandas 和 NumPy，用于使用 Pandas 的 DataFrame 特性来操作数据。每个数据集首先作为一个单独的 Pandas 数据框架导入。列出了列，以及指示行数、内存大小和属性数据类型的元数据。从每个数据集中删除不需要的列。每个数据集中的现有列都被重命名以提高透明度。

例如，在导入预期寿命数据集后，它具有属性:位置、时期、指示器、Dim1、第一个工具提示。这些名字是根据联合国的调查参数设定的。该模型所需的列，即位置、时期和第一个工具提示，分别被重命名为国家、年份和预期寿命，而其余的列被删除。对于其他数据集，遵循类似的程序。

到目前为止，数据帧是分开处理的。因为这些数据集是为不同的年份和不同的国家创建的，所以合并它们是很繁琐的&很耗费内存。因此，所有数据集都选择了 2012 年，因为这一年包含的 null/空值在所有数据集中最少。现在，所有数据集都根据国家名称进行合并，即保留所有数据集中共有的国家，其余的全部删除。这个转换过程消除了几乎所有的空值。还有一些剩余的 null/空值的国家也被删除。得到的清理和转换后的数据集现在有 12 列和 116 行。转换和清理还成功地消除了异常值，这通过散点图得到了证实。最终数据集导出为. csv 文件，以便使用 SPSS 进行进一步分析。

数据集以. csv 文件的形式导入 R Studio。为了便于建模，对数据进行了转置，清除了空值，并在导出到. csv 文件以便在 R Studio 中进一步分析之前对列进行了重命名。该数据集有 2 列，日期和 CPI，288 行，包含从 1996 年 1 月到 2019 年 12 月的月度数据。

# 三。模型论

## A.多元线性回归

该技术用于估计一个因变量和多个自变量之间的关系(Ucla.edu，2019) (Yale.edu，2019)。一个简单的多元线性回归模型

等式看起来像这样(Hyndman 和 Athanasopoulos，2018):

![](img/ae926d0ae788c71c3fcfbde7adb33c4c.png)

要成功实施多元线性回归模型，需要满足一些假设:

1.  响应和每个解释变量之间应该有线性关系。它们之间也应该有一个集体线性关系。
2.  响应变量必须是连续的。解释变量也必须是连续的(只有数字变量)。
3.  响应变量和解释变量之间不应有任何交叉相关或多重共线性。
4.  响应变量和解释变量应该是同方差的。
5.  误差应呈正态分布(近似)。

有一些特定的评估技术用于判断回归模型是否满足上述假设。如果不满足这些假设，则修改模型以克服这一缺陷。使用的标准是(弗罗斯特，2017 年 b)(弗罗斯特，2017 年 c)(杰森·布朗利，2018 年):

*   调整后的 R 平方值
*   皮尔逊相关值
*   p 值检验(方差分析表)
*   方差膨胀因子(VIF)值和系数的 p 值
*   系数相关性
*   正常峰峰值图
*   剩余散点图

## B.时间序列分析

时间序列分析是分析和预测任何给定时间序列数据的重要工具。时间序列可能具有各种特征——季节性、趋势、噪声、静止/移动方面。执行分解技术是为了分离这些特征，并为分析带来清晰度。如果时间序列相对于时间是移动的，那么有区别地使其静止(Ambatipudi，2017)。

然后，通过研究 ACF(自相关函数)和 PACF(偏自相关函数)图并计算 p(自回归阶)、d(差异度)和 q(移动平均阶)值，选择模型来拟合记录值。本报告中使用的模型是 ARIMA(自回归综合移动平均)(2017)。

基于两个测试(2017)检查 ARIMA 模型的准确性:

*   常态 Q-Q 图:检查数据的常态
*   永盒检验:用 p 值检验假设

测试成功后，该模型用于预测所需持续时间的值。该预测模型使用模型与时间图进行测试，并再次使用永盒测试(Statistics Solutions，2017)。

# 四。模型概述和分析

## A.多元线性回归

在 SPSS 中建立了多个模型并进行了测试(Laerd.com，2018) (Statistics Solutions，2017)。根据上一节提出的假设和参数对模型进行了分析。模型摘要包括:

***1。型号-1:***

*模型方程:*

![](img/0991a97832b5740264a6c0d1f078b92e.png)

*调整后的 R2:* 0.969

*皮尔逊相关值:*

*   预期寿命:人类发展指数= 0.886
*   预期寿命:成人死亡率= -0.945
*   脊髓灰质炎 _ 免疫力:乙型肝炎 _ 免疫力= 0.847

*p 值(方差分析表)* : 0.000 ( < 0.005)

*VIF &系数的 p 值:*

*   VIF 值(成人死亡率)= 6.316
*   VIF 值(乙肝免疫)= 5.320
*   p 值(总人口)= 0.538
*   p 值(hepB_immunity) = 0.473
*   p 值(脊髓灰质炎 _ 免疫力)= 0.062
*   p 值(卡介苗免疫)= 0.718

*系数相关值:*

*   所有相关值< 0.8

*正常峰峰值图:*参见图 1

*残差散点图:*参见图 2

*分析:*即使 R2 值是最佳的，皮尔逊相关值也显示出高度的相关性。由于人类发展指数和成人死亡率与因变量预期寿命相关，因此必须剔除。脊髓灰质炎 _ 免疫、乙肝 _ 免疫&卡介苗 _ 免疫具有高度相关性，它们的系数净 p 值也是> 0.05，这意味着不能拒绝零假设。对于下一个模型，这三个变量将通过取平均值合并成一个 avg_immunity 变量。图表示残差和同方差的正态性。人们还注意到，人口和国内生产总值数据值与其他数据值相比非常高，因此对这些变量应用了对数(以 10 为底),以使所有数据具有相似的规模。图 17 中的组合表。 *2。*

**2*。型号-2:***

*模型方程:*

![](img/d36f0dedb32973dcea1274141a4254bd.png)

*调整后的 R2:* 0.822

*皮尔逊相关值:*

*   所有相关值< 0.8
*   p-value (ANOVA table): 0.000 (< 0.005)

*VIF &系数的 p 值:*

*   所有 VIF 值< 5
*   p-value (population) = 0.065

*系数相关值:*

*   所有相关值< 0.8

*正常峰峰值图:*参见图 3

*残差散点图:*参见图 4

*分析:*该模型中唯一需要关注的值是人口变量系数的高 p 值。它是> 0.05，表示未能拒绝零假设，即系数是不重要的。因此，可以移除该变量，而不会显著影响模型。图表示残差和同方差的正态性。图 18 中的组合表。

***3。型号-3:***

*模型方程:*

![](img/8f9669e57059ec1018762373c53bd1d8.png)

*调整后的 R2:* 0.818

*皮尔逊相关值:*

*   所有相关值< 0.8
*   p-value (ANOVA table): 0.000 (< 0.005)

*VIF &系数的 p 值:*

*   所有 VIF 值< 5
*   All p-values < 0.05

*系数相关值:*

*   所有相关值< 0.8

*正常峰峰值图:*参见图 5

*残差散点图:*参考图 6

*分析:*所有值和图都符合设定的指标。图表示残差和同方差的正态性。这个模型是这个项目的最终回归模型。图 19 中的组合表。

## B.时间序列分析

预测模型在 R (Peixeiro，2019) (Doc，未注明日期)中进行编码、绘制和分析。这些步骤如下:

*1。分解:*

在清理之后，时间序列被导入用于分解，以适当地观察组成时间序列的不同元素，即，序列的趋势、序列内的季节性、数据的移动/静止性质以及最终在序列内渗透的噪声，如图 7 所示。该序列是非平稳的季节性序列，具有上升趋势和一些噪声。R 中的自动 ARIMA 函数用于帮助获取模型的 p(自回归阶)、d(差异度)和 q(移动平均阶)值，因为使用 ACF 和 PACF 函数不可能做到这一点。

*2。ARIMA:*

自动 ARIMA 函数建议采用带有漂移的 ARIMA(0，1，2)(0，0，2)[12]模型(参见图 8)。该模型用于检验残差的正态性。如图 9 所示，残差呈正态分布。但是来自 Ljung-Box 测试的 p 值是 0.1464 (>0.05，图 10)。为了对此进行校正，调用了 R 中的差分函数，如图 11 所示，这产生了好得多的 p 值(几乎为零，图 12)。

*3。分析和预测:*

如图 13 所示，绘制了该系列的标准 Q-Q 图。几乎所有的数据点都很好地符合这条线，因此，时间序列是正态分布的。

接下来，该模型用于预测未来两年的 CPI。如图 14 和图 15 所示，模型精确地拟合了观察值，并且预测模型精确地遵循了序列的上升趋势。最后，使用容格盒测试对结果进行检验。

*4。永盒试验:*

该测试的 p 值非常接近于零，如图 16 所示。这意味着模型精确地符合观察值，这是期望的结果。

## C.图表和表格

![](img/6797a3e0f6d6477cf205afe4cad6e859.png)

图一。模型 1:正常峰峰值图

![](img/f7159adf3aa88e7a1f60838decffbad7.png)

图二。模型 1:残差散点图

![](img/abdc49e659e2f8c42c129476050a786f.png)

图 3。模型 2:正常峰峰值图

![](img/662ff9026d10c99ed08a6c3ed1f20d4f.png)

图 4。模型 2:残差散点图

![](img/d33bb6134ac20d52bb689a5c61da00f8.png)

图 5。模型 3:正常峰峰值图

![](img/bf71b3034c1477107e8f918ddcfe4f9a.png)

图 6。模型 3:残差散点图

![](img/cebc594895604c3b29d9f9d8c80716fb.png)

图 7。乘法时间序列的分解图

![](img/9131e4fe7aa093b6714dfa71ff566a96.png)

图 8。最佳模型建议，由 auto.arima()制作

![](img/00bab16ff5276859327932f3683b2587.png)

图 9。含 ACF 的残差图:关于时间序列

![](img/afbfe4b09b7b54f261268f4aa7d75bbe.png)

图 10。auto.arima()给出的时间序列模型的容格检验

![](img/6c11a9000da84dd9f1eab13b257bd8f1.png)

图 11。含 ACF 的残差图:差分时间序列后

![](img/36fdac6cf72a4c1780868d9951ac4d0c.png)

图 12。差分时间序列后的容格检验

![](img/34689e0ee1d8003e661212121fc6d350.png)

图 13。正常 Q-Q 图

![](img/d04bcbe6e47abf0e724d3600f68338ba.png)

图 14。利用时间序列模型预测未来两年的消费物价指数

![](img/fe75ca901195f8f44ef1cfc1c138322b.png)

图 15。使用时间序列模型(差分)预测未来两年的 CPI

![](img/0331803da17430e33876e0ea0050d97d.png)

图 16。最终模型的永盒测试

# 动词 （verb 的缩写）结论

## A.多元线性回归

在 SPSS 中建立了多个模型并进行了测试。选择了满足前面章节提出的所有假设并产生可观的 R2 分数的模型。最终的模型方程是:

![](img/b626232defa7155b89773d2c9fdcf072.png)

该模型通过了正态性和同质性检验。图 20 给出了皮尔逊相关值的总结。图 22 显示了一个类似的图示，它也包括了所有变量的散点图。这些图形是用 Python 制作的。

## B.时间序列分析

时间序列建模选择了具有漂移的 ARIMA(0，1，2)(0，0，2)[12]模型，并对时间序列进行差分以减少误差。虽然正态 Q-Q 图证实了数据表现出正态性，但 Ljung-Box 检验证实了模型是准确的。

# 不及物动词承认

作者要感谢爱尔兰国家学院为本项目提供了必要的资源，并衷心感谢托尼·德莱尼教授(博士)的不断支持和指导。

# 参考

Ambatipudi，V. (2017 年)。SPSS 中的时间序列分析。YouTube。上市时间:[https://www.youtube.com/watch?v=0ew9XMbkgpo](https://www.youtube.com/watch?v=0ew9XMbkgpo)【2020 年 4 月 5 日上市】。

布赖恩，M.F .和切凯蒂，S.G. (1993 年)。作为通货膨胀衡量标准的消费者价格指数。[在线]美国国家经济研究局。可用地点:【https://www.nber.org/papers/w4505 【2020 年 4 月 5 日获取】。

data.un.org(未标明)。UNdata | explorer。[在线]可在 http://data.un.org/Explorer.aspx【2020 年 4 月 5 日访问】。

多尔蒂，a .和范订单，R. (1982 年)。通货膨胀、住房成本和消费者价格指数。《美国经济评论》，[在线] 72(1)，第 154-164 页。上市时间:[https://www.jstor.org/stable/1808582?seq=1](https://www.jstor.org/stable/1808582?seq=1)【2020 年 4 月 5 日上市】。

欧洲委员会。G20 CPI 所有项目—二十国集团—消费者价格指数(prc_ipc_g20)。[在线]见:[https://EC . Europa . eu/Eurostat/cache/metadata/en/PRC _ IPC _ G20 _ esms . htm](https://ec.europa.eu/eurostat/cache/metadata/en/prc_ipc_g20_esms.htm)【2020 年 4 月 5 日获取】。

弗罗斯特 j .(2017 年 a)。检查您的残差图，以确保可靠的回归结果！[在线]吉姆的统计。可在:[https://statisticsbyjim . com/regression/check-residual-plots-regression-analysis/](https://statisticsbyjim.com/regression/check-residual-plots-regression-analysis/)【2020 年 4 月 5 日获取】。

弗罗斯特，j .(2017 年 b)。如何解释回归分析中的 P 值和系数？[在线]吉姆的统计。可在:[https://statisticsbyjim . com/regression/interpret-coefficients-p-values-regression/获取。](https://statisticsbyjim.com/regression/interpret-coefficients-p-values-regression/.)

弗罗斯特 j .(2017 年 c)。回归分析中的多重共线性:问题、检测和解决方案——统计学。[在线]吉姆的统计。可在:[https://statisticsbyjim . com/regression/multi 共线性-in-regression-analysis/。](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/.)

Hyndman，R.J .和 Athanasopoulos，G. (2018 年)。预测:原理与实践。维克的希斯蒙特。:Otexts。

詹姆斯·g·威滕·d·哈斯蒂·t·蒂布希拉尼·r(未注明)。统计学习导论:r。

杰森·布朗利(2018)。Python 中的正态性测试简介。[在线]机器学习掌握。请访问:[https://machine learning mastery . com/a-gentle-introduction-to-normality-tests-in-python/。](https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/.)

Laerd.com(2018)。如何在 SPSS Statistics | Laerd Statistics 中进行多元回归分析？[在线]可从以下网址获得:[https://statistics . laerd . com/SPSS-tutorials/multiple-regression-using-SPSS-statistics . PHP .](https://statistics.laerd.com/spss-tutorials/multiple-regression-using-spss-statistics.php.)

MasumRumi(未注明日期)。泰坦尼克号的统计分析和 ML 工作流程。[在线]kaggle.com。可从以下网址获取:[https://www . ka ggle . com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic # Part-1:--Importing-needly-Libraries-and-datasets](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic#Part-1:-Importing-Necessary-Libraries-and-datasets)【2020 年 4 月 5 日获取】。

Shumway，R.H .和 Stoffer，D.S. (2017 年)。时间序列分析及其应用:附实例。湛:斯普林格。

统计解决方案。(2017).检验 SPSS-Statistics 解决方案中线性回归的假设。[在线]可从以下网址获得:[https://www . statistics solutions . com/testing-assumptions-of-linear-regression-in-SPSS/。](https://www.statisticssolutions.com/testing-assumptions-of-linear-regression-in-spss/.)

Ucla.edu(2019)。回归分析| SPSS 注释输出。【在线】可在:[https://stats . idre . UCLA . edu/SPSS/output/regression-analysis/。](https://stats.idre.ucla.edu/spss/output/regression-analysis/.)

Yale.edu。(2019).多元线性回归。[在线]可在:[http://www.stat.yale.edu/Courses/1997-98/101/linmult.htm.](http://www.stat.yale.edu/Courses/1997-98/101/linmult.htm.)找到

# 附录

![](img/4f373eb727466dcfcf63be844d19ad9a.png)

图 17。MLR 模型-1:完整模型摘要

![](img/ebd6463919f5f9d15eac29ecedb323fc.png)

图 18。MLR 模型-2:完整模型摘要

![](img/c50175af20b72f7e3cfdcac34de8034b.png)

图 20。MLR 模型-3:完整模型摘要

![](img/4527da078184603f42828df2f3979f14.png)

图 21。MLR 模型-3:相关矩阵

![](img/62c45ab6f0145658df07edfed4627f3f.png)

图 22。MLR 模型-3:相关表

![](img/623d38c15e347c3232bd8b5e2eae7a77.png)

图 23。MLR 模型-3:相关散点图

*最初发表于*[T5【https://astrum-imber.blogspot.com】](https://astrum-imber.blogspot.com/2020/07/life-expectancy-prediction-using.html)*。*