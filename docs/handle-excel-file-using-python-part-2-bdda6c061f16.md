# 使用 Python 处理 Excel 文件(第 2 部分)

> 原文：<https://medium.com/analytics-vidhya/handle-excel-file-using-python-part-2-bdda6c061f16?source=collection_archive---------20----------------------->

![](img/92a74536a5c015b6aee2ab098bae4e89.png)

在之前的博客《 [**用 Python 处理 Excel 文件(第一部分)**](https://bqstack.com/b/detail/103/Handle-Excel-file-using-Python-%28Part-1%29) 中，我已经讲述了如何用 Python 创建 Excel 文件，添加工作表，添加数据，查看 Excel 工作表细节等。现在，我们将介绍如何使用 Python 查看 Excel 数据，以及如何在 Excel 表格单元格值上添加样式。所以让我们从上一篇博客离开的地方继续。

**从 Excel 表中读取:**
现在让我们从 Excel 表中读取数据，我们可以用与编写数据相同的方式读取数据:
1)通过使用单元格地址
data1 = sheet['B2']
2)通过提供行号和列号
data2 = sheet.cell(row=1，column=2)
现在让我们使用下面的代码片段从工作表中读取数据:

```
# import openpyxl module as this is required to work with Excel
import openpyxl as exl  

# provide the Excel file name 
filename="testexcel.xlsx"

# Loading the Excel workbook .xlsx file which we have created earlier
workbook = exl.load_workbook(filename)

# Take the reference of the active sheet
sheet = workbook.active

# Get the values
data1 = sheet['A1'] 
data2 = sheet.cell(row=1, column=2)

# print the Excel sheet values
print("data 1 is {} and data 2 is {}".format(data1.value, data2.value))
```

执行上述 Python 代码后，我们可以得到以下响应:

```
data 1 is id and data 2 is name
```

这样我们就可以使用 Python 读取工作表数据。我们可以使用一个循环从 Excel 表中获取完整的表格，并使用 Python 打印出来。

**将一张表的数据复制到另一张表:**
我们可以将一张表的数据复制到新的复制表中。我们可以编写以下代码来生成任何工作表的副本:

```
# import openpyxl module as this is required to work with Excel
import openpyxl as exl  

# provide the Excel file name 
filename="testexcel.xlsx"

# Loading the Excel workbook .xlsx file which we have created earlier
workbook = exl.load_workbook(filename)

# get Sheet which you want to copy
source=workbook['Anurag test sheet']
# create the copy sheet
target=workbook.copy_worksheet(source)

# save workbook
workbook.save(filename)
```

使用上面的代码，我们可以创建一个“Anurag 测试表”表的副本。

**从 Excel 文件中移除工作表:**
现在我们要从 Excel 文件中移除工作表，为此，我们需要通过传递工作表引用来调用 Remove()方法，参见下面的代码片段:

```
# import openpyxl module as this is required to work with Excel
import openpyxl as exl  

# provide the Excel file name 
filename="testexcel.xlsx"

# Loading the Excel workbook .xlsx file which we have created earlier
workbook = exl.load_workbook(filename)

# remove a sheet from the Excel file.
workbook.remove(workbook['Anurag test sheet Copy'])

# save workbook
workbook.save(filename)
```

使用上面的代码，我们可以从 Excel 文件中删除“Anurag 测试表副本”表。

**在 Excel 单元格数据中添加样式:**
我们可以改变字体，填充到单元格中。如果我们想为一个单元格配置字体，那么我们需要首先定义字体，然后将该字体分配给所需的单元格，请参考下面的代码片段:

```
font = Font(color = colors.GREEN, bold = True, italic = True)
cell1.font = font
```

使用上面的代码，我们可以改变特定单元格值的格式。同样，我们也可以在需要设置填充变量的地方更改填充颜色，然后将其分配给任何单元格，请参考下面的代码片段:

```
fill = PatternFill(fill_type = 'lightUp', bgColor = 'D8F1D3')
cell2.fill = fill
```

请参考下面的代码，我们在这里应用字体，并填写我们之前使用的相同的 excel 文件。

```
# importing openpyxl and its styles module as this is required to work with Excel
import openpyxl as exl  
from openpyxl.styles import *

# providing the Excel file name 
filename="testexcel.xlsx"

# Loading the Excel workbook .xlsx file which we have created earlier
workbook = exl.load_workbook(filename)

# Fetching the current sheet
sheet = workbook['Anurag test sheet']

# Fetching the cell values
cell1 = sheet['B2']
cell2 = sheet['C2']

# Configuring the font for the first cell
font = Font(color = colors.GREEN, bold = True, italic = True)
cell1.font = font

# Configuring the fill for the first cell
fill = PatternFill(fill_type = 'lightUp', bgColor = 'D8F1D3')
cell2.fill = fill

# saving the workbook
workbook.save(filename)
```

执行上述代码后，我们可以得到以下输出(参考截图)。

![](img/62521e8cf24e9772bb6ad3364a308f80.png)

**对 Excel 中的表格数据应用样式:**
在上一节中，我们对单个单元格进行了样式化，但是使用 Table 和 TableStyleInfo 模块，我们可以对给定范围的完整表格数据进行样式化。我们首先需要创建一个表格对象，如下面的代码片段所示:

```
table = Table(displayName = "Table", ref = "A1:G11")
```

之后，我们可以创建 styel 对象，如下面的代码片段所示:

```
style = TableStyleInfo(name = "TableStyleMedium9", showRowStripes = True, showColumnStripes = True)
```

然后，我们可以将样式分配给表格:

```
table.tableStyleInfo = style
```

最后，我们可以将表格添加到工作表中:

```
sheet.add_table(table)
```

请参考下面的代码片段，我们可以使用它来设置表格 Excel 数据的样式:

```
# importing openpyxl and its styles module as this is required to work with Excel
import openpyxl as exl  
from openpyxl.styles import *
from openpyxl.worksheet.table import Table, TableStyleInfo

# providing the Excel file name 
filename="testexcel.xlsx"

# Loading the Excel workbook .xlsx file which we have created earlier
workbook = exl.load_workbook(filename)

# Fetching the current sheet
sheet = workbook['Anurag test sheet']

#Creating a table inside the sheet
table = Table(displayName = "Table", ref = "A1:C3")

#Defining a style for the table (default style name, row/column stripes)
#Choose your table style from the default styles of openpyxl
#Just type in openpyxl.worksheet.table.TABLESTYLES in the Python interpreter
style = TableStyleInfo(name = "TableStyleMedium9", showRowStripes = True, showColumnStripes = True)

#Applying the style to the table
table.tableStyleInfo = style

#Adding the newly created table to the sheet
sheet.add_table(table)

# saving the workbook
workbook.save(filename)
```

使用上面的代码，我们可以格式化 Excel 文件中的表格数据。不可能在两篇博客中涵盖 openpyxl 模块的所有特性，但是我已经尝试涵盖了一些重要的特性，通过这些特性，任何人都可以使用 Python 处理 Excel 文件操作。如有任何疑问，请随时留下评论。

*最初发表于*[*【https://bqstack.com】*](https://bqstack.com/b/detail/104)*。*