# 新冠肺炎世界进展—用 Plotly 实现动画数据可视化

> 原文：<https://medium.com/analytics-vidhya/covid-19-world-progression-animated-data-vizualization-with-plotly-1c5dd909822d?source=collection_archive---------9----------------------->

对我来说，隔离意味着学习新的东西，为了了解更多关于新冠肺炎疫情和数据科学的知识，我想调查数据并绘制一些图表。本文解释了所有的工作并展示了结果。

![](img/e5287d1d33f38619cdc803917dcb4fb7.png)

资料来源:freeimages.com

# 获取数据

第一步是找到可用的数据。在做了一些研究后，我使用了来自约翰·霍普斯金大学的关于确诊病例、死亡和康复的数据。我还使用请求包来下载数据，这样我就可以很容易地更新数据。我还需要每个国家的 3 位字母代码，我在这里找到了。

# 预处理

然后，我需要准备数据，将所有数据集整合到一个表中，包含我需要的所有内容。首先，我必须按国家汇总数据，为此，名称需要相同。

```
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

conf = pd.read_csv('data/original/time_series_2019-ncov-Confirmed.csv')
deaths = pd.read_csv('data/original/time_series_2019-ncov-Deaths.csv')

changes = [{'old': 'US', 'new': 'United States of America'},
           {'old': 'Russia', 'new': 'Russian Federation'},
           {'old': 'Venezuela', 'new': 'Venezuela (Bolivarian Republic of)'},
           {'old': 'Bahamas, The', 'new': 'Bahamas'},
           {'old': 'Bolívia', 'new': 'Bolívia (Plurinatioanl State of)'},
           {'old': 'Brunei', 'new': 'Brunei Darussalam'},
           {'old': 'Congo (Brazzaville)', 'new': 'Congo'},
           {'old': 'Congo (Kinshasa)', 'new': 'Congo, Democratic Republic of the'},
           {'old': "Cote d'Ivoire", 'new': "Côte d'Ivoire"},
           {'old': "Gambia, The", 'new': 'Gambia'},
           {'old': "Iran", 'new': 'Iran (Islamic Republic of)'},
           {'old': 'Korea, South', 'new': 'Korea, Republic of'},
           {'old': 'Moldova', 'new': 'Moldova, Republic of'},
           {'old': 'Taiwan*', 'new': 'Taiwan, Province of China'},
           {'old': 'Tanzania', 'new': 'Tanzania, United Republic of'},
           {'old': 'United Kingdom', 'new': 'United Kingdom of Great Britain and Northern Ireland'},
           {'old': 'Vietnam', 'new': 'Viet Nam'}]

for c in changes:
    conf.loc[conf['Country/Region'] == c['old'], 'Country/Region'] = c['new']
    deaths.loc[deaths['Country/Region'] == c['old'], 'Country/Region'] = c['new']
    rec.loc[rec['Country/Region'] == c['old'], 'Country/Region'] = c['new']def calc(data):
    data = data.drop(['Lat', 'Long', 'Province/State'], axis=1)
    data = data.groupby('Country/Region')[list(data.columns[1:])].agg('sum')
    return dataconf = calc(conf)
deaths = calc(deaths)
```

在死亡数据集中，一些国家的数字在一天中不断减少，因此最快的解决方案是将其值等于死亡发生前一天的值。冰岛的数据也需要修正。

```
def preprocess(data):
    for c in range(1, len(data.columns)-1):
        if (data.iloc[:, c] < data.iloc[:, c-1]).sum() > 0:
            print(data.loc[(data.iloc[:, c] < data.iloc[:, c-1]), :].iloc[:, c],
                  data.loc[(data.iloc[:, c] < data.iloc[:, c-1]), :].iloc[:, c-1])
            data.loc[(data.iloc[:, c] < data.iloc[:, c-1]), data.columns[c]] = data.loc[(data.iloc[:, c] < data.iloc[:, c-1]), data.columns[c-1]]
    return data

deaths = preprocess(deaths)# Correcting Iceland data
deaths.loc['Iceland', deaths.loc['Iceland', :] == 5] = 1
```

然后，导入国家代码数据集并连接数据。

```
codes = pd.read_csv('data/original/all.csv').set_index('name')[['alpha-3']]graphDF = codes.join(conf)
```

# 绘制传播图

我想知道的第一件事是新冠肺炎是如何传播到世界各地的，一个很好的图表来理解这一点，将是一个确诊病例和死亡的动画地图。Plotly 是一个重要的工具。我使用 Plotly 内置的世界地图，并通过阿尔法-3 代码来识别国家。

```
graphDF = graphDF.loc[graphDF.iloc[:, 1].dropna().index.values, :]
graphDF['country'] = graphDF.index

graphDF = finalDF.melt(id_vars=['country', 'alpha-3'])
graphDF.columns = ['country', 'alpha-3', 'Day', 'N']# Created hasCases discrete variable
finalDF.loc[:, 'hasCases'] = ((finalDF.N > 0) * 1).astype('str')# Graph code
fig = px.choropleth(finalDF, locations="alpha-3",
                    color='hasCases', 
                    hover_name="Country", # Country to hover info
                    color_discrete_sequence=['gainsboro', 'darkred'], # Chosing the colors of categories
                    title="Countries with Confirmed Cases of COVID-19 arround the World", # Graph title
                    animation_frame='Day',
                    animation_group='Country',
                    template='ggplot2')

fig.update_layout(transition = {'duration': 800})

fig.show()
```

资料来源:约翰·霍普斯金大学

有可能看到亚洲、欧洲和北美是最先暴露于这种病毒的，它刚刚到达南美和非洲。实际上，在第 33 天之后，传播速度似乎加快了。事实上，在非洲和南美洲，在第一例确诊病例后，传播速度似乎加快了。

绘制完图表后，我用 Chart Studio API 轻松地上传了它。

```
cs.tools.set_credentials_file(username='lucasgiutavares', api_key='your_api_key')

cs.tools.set_config_file(world_readable=True,
                         sharing='public')

py.plot(fig, filename='COVID-19 Dissemination', auto_open=True)
```

但是，除此之外，这些国家是如何处理传播的呢？为了回答这个问题，我调查了死亡率(死亡/确诊病例)。

# 绘制死亡率图表

从理论上讲，死亡率较高的国家在处理避免传播、诊断和治疗确诊病例方面可能会有更多的问题，一个动画散点图可以对此提供很好的见解。因此，我再次使用 plotly，用令人惊讶的短代码来做这件事。

首先，我需要合并确诊病例和死亡病例的数据集。

```
conf.loc[:, 'Country'] = conf.index
corrGraphData = codes.join(conf)
corrGraphData = corrGraphData.loc[corrGraphData.iloc[:, 1].dropna().index, :]
corrGraphData = corrGraphData.drop(['alpha-3'], axis=1).melt(id_vars=['Country'], var_name='Day', value_name='Cases')
deaths.loc[:, 'Country'] = deaths.index
corrGraphDeaths = codes.join(deaths)
corrGraphDeaths = corrGraphDeaths.loc[corrGraphDeaths.iloc[:, 1].dropna().index, :]
corrGraphDeaths = corrGraphDeaths.drop(['alpha-3'], axis=1).melt(id_vars=['Country'], var_name='Day', value_name='Deaths')
corrGraphData = corrGraphData.join(corrGraphDeaths[['Deaths']])
corrGraphData.loc[:, 'Death Rate'] = (corrGraphData.Deaths / corrGraphData.Cases).fillna(0)
```

然后，标绘的代码。大小和颜色代表国家的死亡率。

```
# Scatter Death Rate Animated Graph
figScatter = px.scatter(corrGraphData, x='Cases', y='Deaths', size='Death Rate', size_max=40,
                        range_x=[10, 160000],
                        range_y=[1, 4900],
                        hover_name='Country',
                        animation_frame='Day',
                        color='Death Rate',
                        title='COVID-19: Deaths by number of confirmed cases',
                        color_continuous_scale=px.colors.sequential.Reds,
                        range_color=[0, 0.05],
                        log_x=True,
                        log_y=True,
                        trendline='lowess',
                        template='plotly_dark')
fig.update_layout(
    xaxis_title="Confirmed Cases",
    yaxis_title="COnfirmed Deaths",
    )

figScatter.show()
```

上传:

```
cs.tools.set_credentials_file(username='lucasgiutavares', api_key='your_api_key')

cs.tools.set_config_file(world_readable=True,
                         sharing='public')

py.plot(figScatter, filename='COVID-19 Deaths rate', auto_open=True)
```

这就是结果:

资料来源:约翰·霍普斯金大学

人们可能会注意到，当第一批病例发生时，许多国家的死亡率很高。在这一主题中，意大利、伊朗和西班牙的发病率有所下降，但仍有很高的发病率和许多病例。其他国家，如索马里、阿尔及利亚、圣马力诺和菲律宾，病例数量较少，但死亡率较高。另一方面，中国和美国的死亡率大幅下降。此外，一些国家，如俄罗斯、德国、冰岛、新加坡，病例数量不多，到目前为止能够保持较低的死亡率。

一个重要的限制是，随着患病人数的增加，进行了更多的检查，即使没有症状，因此，确诊的病例也更多。因此，很难想象在短期内国家的行动和死亡率下降之间有什么关系。

请随意自己导航！

# 结论

通过一些数据分析，可能会注意到新冠肺炎已经蔓延，但有一些方法来遏制它，这主要取决于每个人，而不仅仅是政府的行动。一些国家比其他国家取得了最好的结果，但做好我们的本分才是最重要的。

此外，重要的是要记住，我不是传染病科学专家，但我真的喜欢从数据中产生知识，所以如果你能通过提供一些见解或纠正来帮助它，请这样做！

因此，如果你喜欢这篇文章，请鼓掌，并随时提出批评和/或见解。呆在家里！