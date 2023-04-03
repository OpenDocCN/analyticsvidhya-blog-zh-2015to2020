# 使用熊猫的一些方法

> 原文：<https://medium.com/analytics-vidhya/some-pythonic-ways-to-use-pandas-and-be-a-faster-warrior-5f341de1d256?source=collection_archive---------24----------------------->

Pandas 是一个强大的库，我们应该尽最大努力在各种可能的情况下利用矢量化。在这里，我想介绍一些场景，在这些场景中，改变方法可以节省处理不断增长数据帧的时间。

![](img/9ff0696bdef70c3a790d7979a1416348.png)

## 我们需要的库:

```
import pandas as pd
import numpy as np
```

# 1.数据集