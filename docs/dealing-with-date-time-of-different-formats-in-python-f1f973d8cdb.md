# 用 Python 处理不同格式的日期时间

> 原文：<https://medium.com/analytics-vidhya/dealing-with-date-time-of-different-formats-in-python-f1f973d8cdb?source=collection_archive---------2----------------------->

我们将从提取当前日期和时间的基本示例开始。

Python 提供了一个模块 **datetime** ，它有一个类 datetime，还有一个方法 now()

**从日期时间导入日期时间**

***使用 now()获取当前日期时间:***

```
*# to get current date and time* dateTimeObj **=** datetime.now()
print(dateTimeObj)**Output:**
2019-12-29 00:57:34.108728
```

***提取日期时间对象的不同元素:***

```
print(f"day: {dateTimeObj.day}, month:{dateTimeObj.month},\
year:{dateTimeObj.year} hour:{dateTimeObj.hour}, \
minute:{dateTimeObj.minute}, second:{dateTimeObj.second}")**Output:**
day: 29, month:12,year:2019 hour:0, minute:41, second:4
```

***只提取当前日期*** 假设我们不想要完整的当前时间戳，我们只对当前日期感兴趣。

```
dateObj **=** datetime.now().date()
print(dateObj)
**Output:**
2019-12-29
```

***只提取当前时间*** 假设我们只需要当前时间，即排除日期部分

```
timeObj = datetime.now().time()
print(timeObj)
**Output:**
00:52:02.925910
```

***将 datetime 对象转换为字符串:*** 我们需要使用 **strftime** (datetime，< format >)将 Datetime 对象转换为字符串

```
datetime_str = dateTimeObj.strftime("%B %d, %Y %A, %H:%M:%S")
print(datetime_str)**Output:**
'December 29, 2019 Sunday, 00:57:34'
```

**将字符串转换为日期时间对象:** 我们需要使用 **strptime** (date_string，< format >)将字符串转换为日期时间对象

```
datetime_str = '28/12/2019 22:35:56'
datetime_obj = datetime.strptime(datetime_str, '%d/%m/%Y %H:%M:%S')
print(datetime_obj)**Output:**
2019-12-28 22:35:56
```

**字符串到目前为止:**

```
date_str = '2019-12-28'
dateObj = datetime.strptime(date_str, '%Y-%m-%d').date()
print(dateObj)**Output:**
2019-12-28
```

**字符串到时间:**

```
time_str = '13:55:26'
timeObj = datetime.strptime(time_str, '%H:%M:%S').time()
print(timeObj)**Output:**
13:55:26
```

**Python 日期/时间格式指令**
%a: Weekday 作为地区的缩写名。sun，mon
%A : Weekday 作为语言环境的全名。星期日，星期一，……
% w:以十进制数表示的工作日，其中 0 表示星期日，6 表示星期六。
%d:以零填充的十进制数表示的一个月中的某一天。
%b:月份作为区域设置的缩写名称。Jan，Feb
%B: Month 作为区域设置的全名。一月
%m:作为零填充十进制数的月份。
%y:没有世纪作为零填充十进制数的年份。
%Y:以世纪为小数的年份。
%H:小时(24 小时制)，用零填充的十进制数。
%I:小时(12 小时制)作为补零的十进制数。
%p:语言环境等同于 AM 或 PM。
%M:以零填充的十进制数表示的分钟。
%S:秒为补零的十进制数。
%f:微秒为十进制数，左边补零。
% z:HHMM[SS]形式的 UTC 偏移量(如果对象是简单的，则为空字符串)。
%Z:时区名称(如果对象是简单的，则为空字符串)。
%j:用零填充的十进制数表示的一年中的某一天。
%U:一年中的周数(星期日是一周的第一天)，以零填充的十进制数表示。第一个星期日之前的所有日子都被认为是第 0 周。
%W:以十进制数表示的一年中的周数(星期一为一周的第一天)。新的一年中第一个星期一之前的所有日子都被视为第 0 周。%c:区域设置的适当日期和时间表示。
%x:区域设置的适当日期表示。
%X:区域设置的适当时间表示。
%%:一个文字“%”字符。