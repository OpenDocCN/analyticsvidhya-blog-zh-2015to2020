# 使用 Python 合并大量文件的 3 个简单步骤

> 原文：<https://medium.com/analytics-vidhya/3-simple-steps-to-merge-a-large-number-of-files-using-python-b673e6b0de95?source=collection_archive---------16----------------------->

![](img/598bb41ee2fc18b389a052d18be2839d.png)

卢卡·布拉沃在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

数据提取后，我们通常需要将这些文件合并在一起，以便进一步分析。可以是几个文件，也可以是几百个文件。只需几个文件就可以很容易地将我们需要的数据复制并粘贴到一个工作表中。然而，当处理大量文件时，手动处理是没有意义的。所以在这里，我要分享一个三步走的方法，我曾经用 Python 把数据放在一起。

# 1.导入库

```
import pandas as pd
import os
```

为了合并文件，我们需要使用两个模块，pandas 用于读取 CSV 文件，os 用于与操作(文件)系统交互。

# 2.定义变量

```
file_list = os.listdir(r'C://Users//kelly//Desktop//data)
Folder_Path = r'C://Users//kelly//Desktop//data'
SaveFile_Path = r'C://Users//kelly//Desktop//data'
SaveFile_Name = r'merged.csv'
```

然后，为了简化代码，我创建了 4 个变量。只有 file_list 的数据类型是列表(文件夹路径下的所有文件)，除此之外都是字符串。

# 3.循环运行

```
for i in range(1, len(file_list)):
    df = pd.read_csv(Folder_Path + '//' + file_list[i])
    df.to_csv(SaveFile_Path + '//' + SaveFile_Name, index = False, header = False)
```

最后，在 for 循环中，我逐个读取文件并保存到另一个文件中(在本例中是 merged.csv)。因此，merged.csv 文件将按行追加。如果需要，还可以更改 to_csv 的参数。

# 关于我

嗨，我是 Kelly，一名具有新闻和传播背景的商业分析研究生，喜欢分享探索数据和有趣发现的生活。如果您有任何问题，请随时联系我，电话:【kelly.szutu@gmail.com】T5