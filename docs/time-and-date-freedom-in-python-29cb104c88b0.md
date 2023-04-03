# Python 中的时间(和日期)自由

> 原文：<https://medium.com/analytics-vidhya/time-and-date-freedom-in-python-29cb104c88b0?source=collection_archive---------7----------------------->

这似乎总是一个时间问题。那是什么时候发生的？从那以后发生了多少？什么时候会这样？

![](img/9cf9a4d092b876856d0ed5cff198b909.png)

但是在阅读 python“how to”时，日期和时间就像是二等公民。在书中，处理日期和时间被埋在最后几章。这是一个简短的提及，就像它是你永远都不需要的东西。现在我确信你已经意识到你确实需要它。让我带你踏上一段旅程，这样你就可以开始体验 python 方式的实时自由。

立即开始——如果您想知道现在几点了:

```
**import** **datetime**

datetime.datetime.today()
```

结果: *datetime.datetime(2020，12，3，17，55，3，361780)*

也就是今天 2020 年 12 月 3 日 17:55:03。你明白了。

不要担心，这不会是一个枯燥的主题，一个方法一个方法的爬行。我有个故事来缓解一下气氛。

假设您正在创建一个生成月度财务报告的系统，它通过调用外部 API(可能是 AWS 的成本管理器)来实现这一点。您必须创建一个类来根据指定的年份和月份产生所需的时间输入。我想你可能需要做以下事情:

*   确定指定月份的开始和结束日期
*   确定指定月份的会计年度
*   确定会计年度的开始和结束日期
*   确定指定月份和会计年度开始之间的月数(如果要创建一些月平均值)

让我们处理第一个用例。

**指定月份的起止日期**

您将需要创建一个返回指定月份的开始和结束日期的类。从测试的角度来看，我将假设“datetime”类型是 return，因为如果我们想要向类添加额外的功能，它会更加灵活。你可以看到上面的“日期时间”格式。因此，测试可能是这样的:

```
test_value1=YearMonth('202011').start_monthdate
test_value2=YearMonth('202011').end_monthdate

**assert** test_value1==datetime.datetime(2020,11,1,0,0,0,0),\
        'FAIL create start date'
**assert** test_value2==datetime.datetime(2020,11,30,0,0,0,0),\
        'FAIL create end date'

print(f'PASSED start and end date specified month\
         **{**test_value1, test_value2**}**')
```

我用的是 Jupyter 笔记本，所以打印比记录结果更方便。我假设月份是用“YYYYMM”格式的字符串标识的。

要获得一个月的结束日期，我们可以使用 calendar 模块中的“monthrange”方法。如果提供了年份和月份，它将返回该月的第一个工作日和该月的天数。第二个值显然是我们需要的。另外，我习惯于使用 dateutil 包中的“parse”方法来解析开始日期。开始日期的日期显然是已知的(“01”)。parse 方法比您可以采用的其他方法更加宽容。

课堂是这样的:

```
**import** **calendar**
**from** **dateutil** **import** parser **as** p

**class** **YearMonth**:
    **def** __init__(self,year_month):

        self.year_month=year_month

        year=year_month[0:4]
        month=year_month[4:]
        *# convert year, month to date for first day of month*
        self.start_monthdate=p.parse(year_month+'01')
        *# convert year, month to date for last day of month*
        self.end_monthdate=p.parse(year_month+str(calendar.\
                            monthrange(2020,int(month))[1]))

        **return**
```

您可以尝试一下“monthrange”方法来了解它。如上所述，它返回的第一个值是该月的第一个工作日，星期一为 0。

**财政年度，开始日期和结束日期**

快速前进——不要浪费时间。有三件事情需要测试，但是在编写测试之前，您可能需要检查数据类型。有一个非常方便的软件包——“fiscal year”($ pip install fiscal year)。它有一个类“FiscalDate ”,带有一个属性“fiscal_year ”,这很有帮助。还有一个带属性的类“FiscalYear”。开始“和”。结束”。为这个包的创建者的清晰命名打满分。

要在编写测试之前稍微了解一下这些类:

```
*# explore date types required to create tests*
**import** **datetime**
**import** **fiscalyear**

fiscalyear.START_MONTH = 7

*# create exploratory datetime based on today*
td=datetime.datetime.today()

*# create exploratory fiscal year*
fy=fiscalyear.FiscalDate(td.year,td.month,td.day).fiscal_year
*# fiscal year and type*
print(fy, type(fy))
```

还给我们: *2021 < class 'int' >*

有必要通过设置“START_MONTH”来定义财政年度开始的月份，在我们的例子中，7 表示 7 月。

如果您想了解财政年度的开始日期，您可以尝试在上面的代码中添加:

```
# create exploratory start date for determined fiscal year
fy_startdate=fiscalyear.FiscalYear(fy).start.date()# start date of fiscal year and type
print(fy_startdate, type(fy_startdate))
```

哪个会给你回:*2020–07–01<class ' datetime . date '>*

现在你知道在你的测试中使用哪种类型了。因此，您的测试可能看起来像这样:

```
test_value1=YearMonth2('202011').fiscal_year
test_value2=YearMonth2('202011').fiscal_startdate
test_value3=YearMonth2('202011').fiscal_enddateassert test_value1==2021, 'FAIL return fiscal year'
assert test_value2==datetime.date(2020,7,1)
assert test_value3==datetime.date(2021,6,30)print(f'PASSED fiscal year, start date, end date \
               {test_value1, test_value2, test_value3}')
```

所以我们已经从探索过程中得到了大部分我们需要的东西。该类可能如下所示:

```
# class
import calendar
from dateutil import parser as p
import fiscalyearfiscalyear.START_MONTH = 7class YearMonth2:
    def __init__(self,year_month):

        self.year_month=year_month

        year=year_month[0:4]
        month=year_month[4:]
        # convert year, month to date for first day of month
        self.start_monthdate=p.parse(year_month+'01')
        # convert year, month to date for last day of month               
        self.end_monthdate=p.parse(year_month+\
              str(calendar.monthrange(2020,int(month))[1]))

        # determine fiscal year
        self.fiscal_year=fiscalyear.FiscalDate\
              (self.start_monthdate.year,
               self.start_monthdate.month,
               self.start_monthdate.day).fiscal_year
        # determine fiscal year start date
           self.fiscal_startdate=fiscalyear.FiscalYear\
                         (self.fiscal_year).start.date()
        self.fiscal_enddate=fiscalyear.FiscalYear\       
                         (self.fiscal_year).end.date()
        return
```

我把它命名为 YearMonth2，这样你就不会覆盖这个类的第一个版本。您可以看到，我将“FiscalDate”类的财政年度定义为“datetime”。这是我们在探索阶段发现的。

**当月与财政年度开始之间的月份**

“dateutil”包中有一个类叫做“relativedelta”。它接受两个日期，并以自己的数据类型“relativedelta”给出它们之间的差异。为了探索这一点，我建议这样做:

```
#explore
from dateutil.relativedelta import relativedeltadiff=relativedelta(datetime.date(2020,11,30),\
                     datetime.date(2020,7,1))print(diff, type(diff))
```

这将返回:

*relativedelta(月=+4，日=+29)*

*<类‘dateutil . relative delta . relative delta’>*

不完全是我们正在寻找的，因为我们想要的答案是五个月。但是在这个阶段我们已经有足够的资料来编写一个测试，所以我的建议是:

```
# test
test_value=YearMonth3('202011').months_betweenassert test_value==5,\ 
  'FAIL MonthsBetween specified month and start of financial year'print(f'PASSED fiscal year, start date, end date {test_value}')
```

我将该类命名为 YearMonth3，这样我们就不必覆盖以前的类。

在这堂课上，我们可以采取两种方法中的一种来处理我们没有得到五个月的事实。我们可以使用指定月份的开始日期并添加一个月，或者我们可以在指定月份的结束日期上添加一天，即下个月的第一天。为了给我们一些额外的练习，我选择了后者，所以这个类看起来像这样:

```
import calendar
from dateutil import parser as p
from dateutil.relativedelta import relativedeltaclass YearMonth3:
    def __init__(self,year_month): self.year_month=year_month
        year=year_month[0:4]
        month=year_month[4:]
        # convert year, month to date for first day of month
        self.start_monthdate=p.parse(year_month+'01')
        # convert year, month to date for last day of month
        self.end_monthdate=p.parse(year_month+str(\
                      calendar.monthrange(2020,int(month))[1])) # determine fiscal year
        self.fiscal_year=fiscalyear.FiscalDate(\
                          self.start_monthdate.year,
                          self.start_monthdate.month,
                          self.start_monthdate.day).fiscal_year
        # determine fiscal year start date
        self.fiscal_startdate=fiscalyear.FiscalYear(\
                               self.fiscal_year).start.date()
        self.fiscal_enddate=fiscalyear.FiscalYear(\
                               self.fiscal_year).end.date()
        #convert year, month to first day of the following month
        self.end_p1_monthdate=self.end_monthdate+\
                                   relativedelta(days=1)
        # months between spefcified month and start of fiscal year
        self.months_between = relativedelta(\
                               self.end_p1_monthdate,
                               self.fiscal_startdate).months return
```

由此可以看出,“relativedelta”可以用来为日期增加时间，在本例中，为我们已经定义的日期增加 1 天。

**结论**

我希望你喜欢我们的时间旅行。我真的认为处理基于时间的数据的例子应该得到更多的重视，并希望我的贡献被认为是有意义的。

如果您觉得这很有帮助，您可能还会对以下内容感兴趣:

 [## 在 Jupyter 笔记本中更快地组织 Boto3 自动化工具原型

### 一点点的组织可以走很长的路。我想和你分享我的一些组织实践…

stuart-heginbotham.medium.com](https://stuart-heginbotham.medium.com/getting-organised-to-more-quickly-prototype-boto3-automation-tools-in-jupyter-notebooks-8c1850e53047)