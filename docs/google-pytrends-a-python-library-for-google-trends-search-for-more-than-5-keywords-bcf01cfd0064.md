# 比较两个以上的列表——Google py Trends——一个用于 Google Trends 搜索超过 5 个关键词的 python 库

> 原文：<https://medium.com/analytics-vidhya/google-pytrends-a-python-library-for-google-trends-search-for-more-than-5-keywords-bcf01cfd0064?source=collection_archive---------14----------------------->

![](img/6c30e5a0922877300f7899eba6a679ec.png)

大家好。这是我在媒体上的第一个故事。我一直在纠结这个问题，所以我想，这将是一个好的开始，让每个人都知道一个更简单的解决方法。所以事不宜迟，让我们开始吧。

每个需要自动化 google trends 部分的人都必须知道 python 的 pytrends 库。我们有很多关于它的相关查询、相关关键词、兴趣等信息。但是如果有 5 个以上的关键字，我们仍然需要从所有的关键字中找到一个搜索最多的。

所以，让我们从一个列表中有 10 个关键字开始。我们将开始分组前 5 个，并找到一个中等排名的关键字。此处的链接将帮助您更好地理解谷歌趋势搜索如何处理 5 个以上的关键词。[https://digitaljobstobedone . com/2017/07/10/how-do-you-compare-large-number-of-items-in-Google-trends/](https://digitaljobstobedone.com/2017/07/10/how-do-you-compare-large-numbers-of-items-in-google-trends/)。

现在，我们正在做的是自动化这整个部分，只是为了达到最后最搜索的关键字的结果的结论。最初，第一组将由 5 个关键词组成，然后列表将包含不到 5 个关键词和前一组中排名中等的关键词。让我们从代码的第一部分开始。

```
**def** keywords_more_than_5(searches):#searches is the list with more than 5 keywords
#kw is an empty list to create a new list of 5 or less than 5 keywords
    i = 0interest_over_time_df = {}#count increases according to the number of groups and that number of dictionaries will be created     count = 0
    flag=0
    **while** i < len(searches) :
        kw = []
        **if** i < len(searches):
            kw.append(searches[i])
        **if** i + 1 < len(searches):
            kw.append(searches[i + 1])
        **if** i + 2 < len(searches):
            kw.append(searches[i + 2])
        **if** i + 3 < len(searches):
            kw.append(searches[i + 3])
        **if** i + 4 < len(searches):
            kw.append(searches[i + 4])
        flag=1
        i=i+5 **i**nterest_over_time_df[count] = list_pytrends(kw)
```

在上面的代码中，keywords_more_than_5 是一个接受搜索的函数——一个超过 5 个关键字的列表作为它的参数。i=0 是搜索列表的索引。interest_over_time_df 是一个新的字典，计数将跟踪将要创建的组和字典的数量。最初标志=0，这将有助于我们在某个时候进行平均缩放。现在，我们将为 5 个关键字创建一个空间，并根据列表长度检查每个索引。对于第一组，当 i=0 时，列表的第 4 个索引将被认为是第 5 个关键字，并且列表 kw 将具有其中的所有 5 个元素。我们已经完成了第一组的创建。现在下一步是将这个列表发送到 pytrends，正如我们在函数 list_pytrends 上面看到的。

```
**def list_**pytrends(kw):
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=kw, geo=**'AE'**, timeframe=**'today 12-m'**, cat=124)
    interest_over_time_df = pytrend.interest_over_time()
    interest_over_time_df = interest_over_time_df.drop(**'isPartial'**, axis=1)
    **return** interest_over_time_df
```

这个函数提供了 pytrends 的所有必需品，比如列表、位置、时间表和类别。这个链接是针对所有类别的，你可以在你的代码中嵌入这些类别来找到基于特定类别的完美结果。[https://github . com/pat 310/Google-Trends-API/wiki/Google-Trends-Categories](https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories)。现在，我们知道了调用内置函数 interest_over_time()的任务，该函数将返回 pandas.Dataframe。现在，该 Dataframe 将被视为第一组关键字，这将帮助我们根据第一组中所有关键字的平均值找到中等排名的关键字。

```
#find only the middle brand until the last group**if** kw[len(kw)-2]!=searches[len(searches)-1]:
        middle_brand = middle_product(interest_over_time_df[count])

interest_over_time_df[count].loc[**'mean'**]=interest_over_time_df[count].mean(0)
       **if** flag==1:
               interest_over_time_df[count].loc[**'scaling'**]=interest_over_time_df[count].mean(0)
 count += 1
```

现在是时候从数据帧中找到中间产品，并向数据帧添加一个均值行和一个缩放行，我将在后面解释。对于第一组，缩放将保持与平均行相同，因此旗帜进入图片。

```
**def** middle_product(interest_over_time_df):
    avg_list = interest_over_time_df.mean(0)
    x = {k: v **for** k, v **in** sorted(avg_list.items(), key=**lambda** item: item[1])}
    j = 0
    **for** key **in** x:
        **if** (j == 2):
            middle_brand = (key)
            avg_value=x[key]
        j = j + 1

    **return** middle_brand,avg_value
```

现在为了找到中间产品，我们有一个字典 avg_list，由 python 中 axis=0 的均值函数组成。我们根据平均值对字典进行排序，然后选择中间品牌及其对应的平均值，这将有助于我们稍后找到缩放行。

```
if i!=0:
   flag=0
   kw.append(middle_brand[0])
   prev_middle_brand=middle_brand[1]
   i = i + 4
```

张贴这个，我们将在 I 中有增量，我们将移动到下一个组，这将导致我们添加接下来的 4 个关键字和中间排名关键字。我们将保存中间关键字的平均值，以找到缩放值。同样，重复的任务将通过找到数据帧来完成，该数据帧将具有其均值行和附加到其上的缩放行。但是现在缩放行将与第一组不同。

```
interest_over_time_df[count].loc[**'scaling'**]=scaling_func(interest_over_time_df[count],prev_middle_brand) 
```

为了将产品作为一个整体进行比较，我们在下一组中添加了相同的关键字，但我们将根据什么来比较它们，两组的平均值是不同的，因此通过对它们进行缩放，我们将在一个基础上对它们进行比较。把新组的所有平均值乘以上一组的中间积的平均值，再除以它自己组的平均值，就可以做一些很好的分析了。因此，缩放函数什么也不做，只是为我们提供缩放后的值。

```
**def** scaling_func(df,avg_val_prev):
    scaling_list=df.values[-1].tolist()
    common=scaling_list[len(scaling_list)-1]
    list=[]
    **for avg_value in** scaling_list:
        avg_value=avg_value*avg_val_prev/common
        list.append(avg_value)
    **return** list
```

缩放列表将具有附加到数据帧的平均值行中的平均值，并且应用相同的方法，我们将具有包含缩放值的列表，该列表将实际用于比较产品之间的共同点，这样继续下去，我们将具有尽可能多的组，它们可以用 5 个或少于 5 个关键字来创建。

```
df = pd.concat(interest_over_time_df, axis=1)
df.columns = df.columns.droplevel(0)
create_excel(df,searches,string,**""**)
scaling_list=df.values[-1].tolist()
keyword_list=list(df.columns.values)
dict={}
keyword_list=keyword_list[1:]
scaling_list=scaling_list[1:]
**for** i **in** range(0,len(keyword_list)):
    dict[keyword_list[i]]=scaling_list[i]
x = {k: v **for** k, v **in** sorted(dict.items(), key=**lambda** item: item[1],reverse=**True**)}
list_x=[]
**for** key **in** x:
    list_x.append(key)
keywords_less_than_5(list_x[0:5],string)
```

现在是将所有数据帧组合在一起的时候了，所以 pd.concat 也在做同样的事情。将分析视为一个整体将是一个伟大的想法，所以我创建了一个函数 create_excel。现在，我们将所有的缩放值和关键字一起放在一个字典中，我们将按降序排列它们，并拥有我们自己的 5 个在谷歌上搜索最多的关键字。为了得到最终结果，我们将在一个名为 keywords_less_than_5 的函数中发送最终列表作为参数。

```
**def** keywords_less_than_5(name_list):
     interest_over_time_df = list_pytrends(name_list)
     create_graphs(interest_over_time_df)
     create_excel(interest_over_time_df,name_list)
```

我们已经有了所有关键字的最终数据框架，现在是时候创建一个最终的 excel 表格和一个最终的图表，显示其中搜索次数最多的关键字。

```
**def** create_graphs(interest_over_time_df,string):
    sns.set(color_codes=**False**)
    ax = interest_over_time_df.plot.line(figsize=(9, 6),title=**"Interest Over Time"**)
    ax.set_xlabel(**'Date'**)
    ax.set_ylabel(**'Trends Index'**)
    ax.tick_params(axis=**'both'**, which=**'major'**, labelsize=13)
    ax.figure.savefig(**'static/plot_'**+string+**'.png'**)

**def** create_excel(interest_over_time_df,name_list,string,final):
    interest_over_time_df.reset_index(level=0, inplace=**True**)
    pd.melt(interest_over_time_df, id_vars=**'date'**, value_vars=name_list)
     interest_over_time_df.to_excel(**'static/trends_'**+string+**'_'**+final+**'.xlsx'**)
```

因此，它以一个漂亮的图表和一个完美的最终关键字 excel 表结束。

最终代码如下所示

```
**def** keywords_more_than_5(searches,string):
    i = 0
    middle_brand = **""** interest_over_time_df = {}
    count = 0
    flag=0
    **while** i < len(searches) :
        kw = []
        **if** i < len(searches):
            kw.append(searches[i])
        **if** i + 1 < len(searches):
            kw.append(searches[i + 1])
        **if** i + 2 < len(searches):
            kw.append(searches[i + 2])
        **if** i + 3 < len(searches):
            kw.append(searches[i + 3])
        **if** i == 0:
            **if** i + 4 < len(searches):
                kw.append(searches[i + 4])
            flag=1
            i=i+5
        **else**:
            flag=0
            kw.append(middle_brand[0])
            prev_middle_brand=middle_brand[1]
            i = i + 4
        interest_over_time_df[count] = list_pytrends(kw)
        **if** kw[len(kw)-2]!=searches[len(searches)-1]:
            middle_brand = middle_product(interest_over_time_df[count])

        interest_over_time_df[count].loc[**'mean'**] = interest_over_time_df[count].mean(0)
        **if** flag==1:
            interest_over_time_df[count].loc[**'scaling'**]=interest_over_time_df[count].mean(0)
        **else**:
           interest_over_time_df[count].loc[**'scaling'**]=scaling_func(interest_over_time_df[count],prev_middle_brand)
        count += 1
    df = pd.concat(interest_over_time_df, axis=1)
    df.columns = df.columns.droplevel(0)
    create_excel(df,searches,string,**""**)
    scaling_list=df.values[-1].tolist()
    keyword_list=list(df.columns.values)
    dict={}
    keyword_list=keyword_list[1:]
    scaling_list=scaling_list[1:]
    **for** i **in** range(0,len(keyword_list)):
        dict[keyword_list[i]]=scaling_list[i]
    x = {k: v **for** k, v **in** sorted(dict.items(), key=**lambda** item: item[1],reverse=**True**)}
    list_x=[]
    **for** key **in** x:
        list_x.append(key)

    keywords_less_than_5(list_x[0:5],string)**def** middle_product(interest_over_time_df):
    avg_list = interest_over_time_df.mean(0)

    x = {k: v **for** k, v **in** sorted(avg_list.items(), key=**lambda** item: item[1])}
    j = 0
    **for** key **in** x:
        **if** (j == 2):
            middle_brand = (key)
            avg_value=x[key]
        j = j + 1

    **return** middle_brand,avg_value

**def** scaling_func(df,avg_val_prev):
    scaling_list=df.values[-1].tolist()
    common=scaling_list[len(scaling_list)-1]
    list=[]
    **for avg_value in** scaling_list:
        avg_value=avg_value*avg_val_prev/common
        list.append(avg_value)
    **return** list**def** keywords_less_than_5(name_list):
     interest_over_time_df = list_pytrends(name_list)
     create_graphs(interest_over_time_df)
     create_excel(interest_over_time_df,name_list)**def** create_graphs(interest_over_time_df,string):
    sns.set(color_codes=**False**)
    ax = interest_over_time_df.plot.line(figsize=(9, 6),title=**"Interest Over Time"**)
    ax.set_xlabel(**'Date'**)
    ax.set_ylabel(**'Trends Index'**)
    ax.tick_params(axis=**'both'**, which=**'major'**, labelsize=13)
    ax.figure.savefig(**'static/plot_'**+string+**'.png'**)

**def** create_excel(interest_over_time_df,name_list,string,final):
    interest_over_time_df.reset_index(level=0, inplace=**True**)
    pd.melt(interest_over_time_df, id_vars=**'date'**, value_vars=name_list)
     interest_over_time_df.to_excel(**'static/trends_'**+string+**'_'**+final+**'.xlsx'**)
```

希望有帮助:-)。感谢您的宝贵时间！！