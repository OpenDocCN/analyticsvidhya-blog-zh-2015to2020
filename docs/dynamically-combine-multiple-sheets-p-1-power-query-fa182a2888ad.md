# 动态组合多张图纸 p# 1:电源查询

> 原文：<https://medium.com/analytics-vidhya/dynamically-combine-multiple-sheets-p-1-power-query-fa182a2888ad?source=collection_archive---------10----------------------->

![](img/ddced4fae32537a3337bba22f7d3c32f.png)

在开始分析数据之前，分析师的关键任务之一是合并和转换数据。在这种情况下，Power Query 非常方便。

# 案例:当源数据与输出表位于不同的工作簿中时

基本的假设是所有的工作表都有相同的数据表结构。

对于与输出表相同的源表，请参考文章:

[](https://www.vivran.in/post/dynamically-append-multiple-sheets-p-2-power-query) [## 动态追加多张图纸 p# 2:超级查询

### 对于不同于输出表的源表，请参考文章:动态组合多个表 p#1 数据是…

www.vivran.in](https://www.vivran.in/post/dynamically-append-multiple-sheets-p-2-power-query) 

# 示例数据表

数据在四张不同的表上。

![](img/102499e8f576572bcfeba1cb8b21802d.png)

[下载样本数据](https://vivranin-my.sharepoint.com/:x:/g/personal/vivek_ranjan_vivran_in/EU8-J9vE0uZLnWJ2e2EUOWoBDWC5etkivobt9pEWBSsvqw?e=n2VRUJ)

# 第一步:获取数据

数据>获取数据>从文件>从工作簿

![](img/7091148bf9c848bf193764113413ac4b.png)

选择一个工作表>转换数据

![](img/c06182c0299b4d8e138074e4b1a3d7cc.png)

# 步骤 2:删除除源代码之外的所有步骤

![](img/78ac6b287521ac3e2400ee2ff5b81cd8.png)

# 步骤 3:右键单击数据列>删除其他列

![](img/3595d3ade95f424d796e65d8dfcc78c9.png)

# 步骤 4:展开数据列。

![](img/f110746eb8ed62ac9bd94fca29f09c73.png)

# 第五步:调整标题

主页>使用第一行作为标题

![](img/8f31353103e9066d3ee3efac92e03458.png)

过滤掉剩余的标题行。

右键单击>过滤器>不等于

![](img/492b34fb918bf2a82afbced39dd0ef17.png)

# 结果

![](img/2ca6019233a20aa67dc408398f1e9ea2.png)

# 包括工作表名称

如果我们需要在最终输出中包含工作表名称，请在步骤 3 中进行以下调整。

选择名称+数据列>右键单击>删除其他列

![](img/5d33ec9fe4070fe62925098abdbaf201.png)

# 动态输出

将查询输出加载到数据透视表中。

主页>关闭并加载到>数据透视表

![](img/df36074acca1ea636031a0084de4fe60.png)

工作簿中的任何更改，Power Query 都会相应地更新输出。

![](img/d0bda3354c1efa6108936f5e1b16c5c1.png)[](https://www.vivran.in/post/dynamically-append-multiple-sheets-p-2-power-query) [## 动态追加多张图纸 p# 2:超级查询

### 对于不同于输出表的源表，请参考文章:动态组合多个表 p#1 数据是…

www.vivran.in](https://www.vivran.in/post/dynamically-append-multiple-sheets-p-2-power-query) [](https://www.vivran.in/post/combining-multiple-files-in-a-folder-power-query) [## 将多个文件合并到一个文件夹中:超级查询

### 想象一下:有一个 2019 年的文件夹。一年中的每个月有 12 个文件。所有的文件都有…

www.vivran.in](https://www.vivran.in/post/combining-multiple-files-in-a-folder-power-query) 

*我写关于*[*MS Excel*](https://www.vivran.in/my-blog/categories/excel)*[*权力查询*](https://www.vivran.in/my-blog/categories/powerquery)*[*权力毕*](https://www.vivran.in/my-blog/categories/powerbi)*[*权力中枢*](https://www.vivran.in/my-blog/categories/power-pivot)*[*DAX*](https://www.vivran.in/my-blog/categories/dax)*[*数据分析【数据*](https://www.vivran.in/my-blog/categories/data-analytics)*****

**[@imVivRan](https://twitter.com/imvivran)**