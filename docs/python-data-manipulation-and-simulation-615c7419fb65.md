# Python:数据操作和模拟

> 原文：<https://medium.com/analytics-vidhya/python-data-manipulation-and-simulation-615c7419fb65?source=collection_archive---------12----------------------->

在本帖中，我们将介绍使用 Python 进行数据操作和模拟的基本技术。

**数据操作**

A.群体聚集

给定一个数据框架，假设我们想要按不同性别汇总客户订单，我们可以使用一个简单的 groupby 和 agg 函数:

```
values_gender = csv_file\
                       .groupby(['gender']) \
                       .agg(avg_order_values=('value','mean'),\
                            count_order=('value', 'size')) \
                       .reset_index()
```

在代码中，我们从原始数据 csv_file 创建了一个新的数据框 values_gender。Groupby 指定汇总哪个变量，类似于 SQL 中的“group by”。Agg 允许您指定在性别级别汇总哪些变量，以及使用哪些汇总统计数据。在本例中，我们创建了一个新列 avg_order_values，作为每个性别的订单值的平均值；我们还创建了一个列 count_order，作为每个性别的订单数。最后一行 reset_index()将确保性别不被定义为新数据框中的索引。我使用反斜杠告诉 Python 下一行属于同一个命令。

B.λ函数

Lambda 函数允许我们编写简单的短函数，通常与 Apply 函数一起使用。例如，假设我们想要将性别表示从“男性”和“女性”替换为“M”和“f”。我们使用 if 条件:

```
values_gender['gender'] = values_gender['gender']\
                          .apply(lambda x:\
                          "F" if x=="Female" else "M")
```

在上述示例中，x 是数据框 Values_Gender 中的性别列。我们告诉 Python 将性别列(x)中的“女性”替换为“F”；把所有不属于“女性”的都换成“m”

C.Merge(相当于 SQL 中的“Join”)

merge 语句的 Python 语法类似于 join 语句的 SQL 语法。例如，假设我们希望将 A 部分中创建的 Avg_Order_Values 和 Count_Order 列与原始 csv_file 数据帧合并:

```
csv_file.merge(\
        values_gender[['gender','avg_order_values','count_order']],\      
        left_on='gender', right_on='gender', how = 'left')\       
        [['datetime','gender']]
```

在上述 merge 语句中，我们将数据框 Values_Gender 中的 Average_Order_Values 和 Count_Order 列添加到数据框 csv_file 中。在这种情况下，所有“男性”将具有相同的 Average_Order_Values 和 Count_Order，因为我们使用左连接(在“how”子句中)并保留左数据集 csv_file 中的所有行。我们可以用 SQL 重写相同的代码，如下所示:

```
SELECT datetime, gender 
FROM csv_file a
LEFT JOIN (SELECT gender, avg_order_values, count_order 
           FROM values_gender) b
ON a.gender = b.gender
```

有关 Python 和 SQL 之间的更多翻译，请参见:[https://medium . com/jbnetcodes/how-to-rewrite-your-SQL-queries-in-pandas-and-more-149d 341 fc 53 e](/jbennetcodes/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e)

**数据模拟**

模拟在市场营销中很受欢迎。我们需要基本的概率理论来做出假设，并决定如何模拟数据。假设我们想模拟一个客户一生中会下多少订单。

假设客户的交易数量遵循泊松分布，对于某个常数 c，其期望值为λ= c。泊松分布对于计数中的不确定性建模非常有用。它带有以下假设:

a.每个事务都是相互独立的(不是顺序相关的)。这是一个很难的假设，因为顾客可能会集体购买(使用优惠券)，或者你的商店可能会出现在当地报纸上。

b.常数λ—每个时间段的平均事务不变；交易的变化也不随时间而变化。如果您使用“天”作为单位时间间隔，那么您商店的交易数量可能会随着时间的推移而变化，因为在晚餐时间您可能会有更多的顾客。然而，很难选择一个单位时间，因为即使你选择“月”，你仍然要考虑季节性的影响。

c.两个交易不会同时发生。这使我们有机会把每一个事务都看作一个伯努利试验。此外，当样本量无限大而成功概率无限小时，泊松近似二项式分布。实际上，我们很少知道二项式模拟所需的样本大小或成功概率；然而，从我们在单位周期内得到的交易数量中，得到预期比率λ的估计值并不太困难。

取λ= 4，我们模拟:

```
from scipy.stats import poisson
csv_file['predicted_transactions'] = np.random.poisson(4)
```

有关泊松的更多信息，请参见[https://towards data science . com/the-Poisson-distribution-and-Poisson-process-explained-4 e2cb 17d 459](https://towardsdatascience.com/the-poisson-distribution-and-poisson-process-explained-4e2cb17d459)

[https://towards data science . com/poisson-distribution-intuition-and-derivation-1059 aeab 90d](https://towardsdatascience.com/poisson-distribution-intuition-and-derivation-1059aeab90d)

实际上，我们可能不想对泊松中λ的值做一个可靠的猜测。为了模拟 lambda，我们可以使用 gamma 分布，其中随机变量始终为正且向右倾斜。后面会讲到伽马！

如果你喜欢我写的东西，请记得发个拍手！