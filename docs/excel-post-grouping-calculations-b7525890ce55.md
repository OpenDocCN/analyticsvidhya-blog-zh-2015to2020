# Excel 分组后计算

> 原文：<https://medium.com/analytics-vidhya/excel-post-grouping-calculations-b7525890ce55?source=collection_archive---------43----------------------->

介绍一个简单的代码来处理 Excel 分组的难题

![](img/ee94d1436536dc8fe75d2131491a5fce.png)

照片由 [Pexels](https://www.pexels.com/photo/person-holding-blue-and-clear-ballpoint-pen-590022/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 的 [Lukas](https://www.pexels.com/@goumbik?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 拍摄

Excel 支持对数据进行分组和聚合，并提供数据透视表来执行多维分析。但是 Excel 的方法和操作过于简单，无法处理复杂的任务或方便地管理它们。这些任务包括获取每个子集的排名，以及按集合进行过滤和排序。在这篇文章中，我们将给出后分组计算的例子，分析它，并提供 SPL 代码解决方案。SPL(结构化过程语言)是专用数据计算引擎 esProc 使用的语言。用它来处理结构化计算很方便。

# 一、组内排序

下面是学生成绩表。我们想得到每个科目的排名。为此，我们需要首先按主题对记录进行分组，然后对每个组进行排序。

下面是 Excel 数据:

![](img/8f70ecc51dc35c58e00cce91de6b3051.png)

预期结果:

![](img/ce769202f009be2c7838870291a37aa7.png)

SPL 脚本通过剪贴板与 Excel 交互。我们在 esProc designer 中编辑一个 SPL 脚本，将待分析的 Excel 数据粘贴到剪贴板，执行脚本，然后将结果粘贴到 Excel。

SPL 以一种简单的方式提供逐步编码:

![](img/d736348a30c128602e4cf70bdf35c700.png)

然后，我们只需将结果粘贴到 Excel 中，以获得所需的表格。

要获得仅包含每个科目前 3 分的记录，我们只需使用 top 函数从每个组中获得合格的记录。

预期结果:

![](img/c05ff76707b79d84f5ead3d1a407a2a6.png)

SPL 使用 top 函数获得前 3 名得分的记录:

![](img/98958e7176689d99c13eadcc92e26284.png)

然后，我们只需将结果粘贴到 Excel 中，以获得所需的表格。

# II 按次骨料分类

另一个场景是根据子集合对组进行排序，例如根据学生分数表，根据学生的总分数对学生进行排名。预期结果如下:

![](img/3a908fee7773408cd8f1878b851de2e1.png)

为此，SPL 使用 groups 函数对记录进行分组，同时计算每个组的总数:

![](img/c98516a09bd8b7cdc920f60b0d936549.png)

# III 组内过滤

这次要找成绩不理想的同学。由于各科考试的难度不同，以不及格来衡量考试是不合适的。相反，我们需要找到那些每科分数低于平均水平的学生。

预期结果:

![](img/c348a3fb58c1f8c7161c5f02492ff9bd.png)

于是我们把记录按科目分组，计算各科平均分，找出成绩低于平均分的学生。

SPL 剧本:

![](img/843fd2700f9defeced50f503a5ece79f.png)

# IV 按子集合过滤

在另一个场景中，我们希望按姓名对记录进行分组，计算每组的总分，并查找总分低于平均总分的学生。

预期结果:

![](img/334e1ee35b899543850567e16d4b1d0d.png)

为了执行一个操作，过滤这里的小计，我们使用 groups 函数来分组记录，计算小计，得到平均总数，并找到符合条件的学生。

SPL 剧本:

![](img/ed6657e210f33641e13c9498f61848e3.png)

# 五.组内百分比计算

以下 Excel 表格存储了 2019 年部分国家的 GDP 值。任务是按洲对记录进行分组，并计算每个国家占其洲 GDP 的百分比。

以下是 2019 年 GDP 表(单位:十亿美元):

![](img/30259511a009ff6fb074f0e4a2790e35.png)

预期结果:

![](img/3d467cb17fe944afdc79983ff4fe7a2d.png)

我们可以将记录按区域分组，将各大洲的 GDP 总量相加，然后计算每个国家的 GDP 在每个组中所占的百分比。

SPL 剧本:

![](img/b3b062336c3fafafc02f46f864dc6910.png)

# VI 计算每组的百分比

我们来看看如何计算各大洲的 GDP 占全球总量的百分比。

预期结果:

![](img/becfeba8eb6406a5b5fb885d4ab5e3ae.png)

这里我们还使用 groups 函数对记录进行分组，计算每个大洲的 GDP，然后得到每个大洲的 GDP 在全球总量中所占的百分比。

SPL 剧本:

![](img/23431fd4bd19e4e6d072e0c38f93de19.png)