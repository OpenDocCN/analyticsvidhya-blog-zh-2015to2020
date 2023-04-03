# 使用 dplyr 包在 R 中进行数据操作

> 原文：<https://medium.com/analytics-vidhya/data-manipulation-in-r-using-dplyr-package-15aeef192208?source=collection_archive---------9----------------------->

![](img/304499d4ffe5c15568b8d5a032105c5f.png)

米卡·鲍梅斯特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

**什么是 dplyr？dplyr 是一个强大的 R 包，用于转换、汇总和执行数据操作。该包包含一组函数(或“动词”)，这些函数执行常见的数据操作，如筛选行、选择特定列、对行重新排序、添加新列和汇总数据。
此外，dplyr 包含一个有用的功能来执行另一个常见任务，即“*拆分-应用-组合* e”概念。**

**重要的 dplyr 动词** 
select():选择列
filter():过滤行
arrange():重新排序或排列行
mutate():创建新列
summary():汇总值
group_by():允许“拆分-应用-组合”概念中的分组操作

*#让我们使用哺乳动物睡眠数据*
从[这里](https://raw.githubusercontent.com/genomicsclass/dagdata/master/inst/extdata/msleep_ggplot2.csv)下载 CSV 格式的 msleep 数据集，然后加载到 R

```
*# Lets first load libraries*
library(dplyr)
msleep <- read.csv(msleep.csv)**#Select verb:** # select name, sleep_total columns
**msleep %>% select(name, sleep_total)**
```

**管子工** : % > %。dplyr 从另一个包(magrittr)中导入这个操作符。该运算符允许您将一个函数的输出通过管道传输到另一个函数的输入，而不是嵌套函数。

```
*# To select all the columns except a specific column, use the “-“ (subtraction) # operator (also known as negative indexing)*
**msleep %>% select(-c(name, genus))***# To select a range of columns by name, use the “:” (colon) operator* **msleep %>% select(name: order)***# To select all columns that start with the character string “sl”, use the 
# function starts_with()*
**msleep %>% select(starts_with(match=”sl”))***# To select all columns that ends with “wt”*
**msleep %>% select(ends_with(“wt”))**
```

**过滤动词:**

```
*# Selecting rows using filter()
# Filter the rows for mammals that sleep a total of more than 16 hours.*
**msleep %>% filter(sleep_total >= 16)***# Filter the rows for mammals that sleep a total of more than 16 hours and have # a body weight of greater than 1 kilogram.*
**msleep %>% filter(sleep_total >=16, bodywt > 1)***# Filter rows for mammals in the Perissodactyla and Primates taxonomic order*
**msleep %>%
 filter(order %in% c(“Perissodactyla”, “Primates”))**
```

**安排动词:**

```
***#****To arrange (or re-order) rows by a particular column such as the taxonomic #order, list the name of the column you want to arrange the rows by* **msleep %>% arrange(order)***# Now, we will select three columns from msleep, arrange the rows by the #taxonomic order and then arrange the rows by sleep_total in desc order. Finally #show the head of the final data frame* **msleep %>%
 select(name, sleep_total, order) %>%
 arrange(order, desc(sleep_total)) %>% head(5)**
```

**变异动词:**

```
*# Create new columns using mutate()
# The mutate() function will add new columns to the data frame. Create a new #column called rem_proportion which is the ratio of rem sleep to total amount of #sleep.*
**msleep %>%
 mutate(rem_proportion = sleep_rem/ sleep_total)**
```

**总结:**

```
*# Create summaries of the data frame using summarise()
# to compute the average number of hours of sleep, apply the mean() function to #the column sleep_total and call the summary value avg_sleep.* **msleep %>%
 summarise(avg_sleep = mean(sleep_total),
 min_sleep = min(sleep_total),
 max_sleep = max(sleep_total))**
```

**使用 group_by()** 分组操作 group_by()动词是 dplyr 中的一个重要函数。
这与“拆分-应用-合并”的概念有关。我们实际上想要通过一些变量(例如分类顺序)分割数据框，将一个函数应用到各个组，然后组合输出。

```
*# Let’s do that: split the msleep data frame by the taxonomic order, then ask for #the same summary statistics as above. We expect a set of summary statistics #for each taxonomic order.*
**msleep %>%
 group_by(order) %>%
 summarise(max_sleep = max(sleep_total)) %>%
 arrange(desc(max_sleep))**
```