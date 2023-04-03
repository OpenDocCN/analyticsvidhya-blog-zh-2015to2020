# 基于配置单元文件的单元测试

> 原文：<https://medium.com/analytics-vidhya/hive-unit-testing-md-6648c9f3b78?source=collection_archive---------17----------------------->

![](img/eaf0498bb5bfd9f2771a6c4f64cad198.png)

测试对编码真的很重要。它让代码确实工作。这确实是一种节省时间的方法，而不是像人们认为的那样浪费时间。但是我绝对不会忽视单元测试是编码阶段中最困难的事情之一这一事实。如果是处理数据，那就更难了。许多人放弃了那个具有挑战性的阶段。如果你关心你所交付的工作的质量，并且确保你所交付的东西你认为是正常工作的，你就必须投入进去。犯错是这项工作的本质。这几乎与成为一名优秀的开发人员无关。一个好的开发者也会犯一些简单的错误。我们不是机器。但是，让我们成为糟糕的开发人员的不是使用机器来测试和确保我们的工作真正完成，我的意思是单元测试。让我们把它放在一边，因为这是一个应该在另一篇文章中讨论的话题。

正如我所说的，当你处理测试需求数据时，很难准备那些被称为已提供数据或初始数据的测试数据，有两种方法可以提供给你的单元测试。一种是我们称之为**的基于文件的**，您可以在 CSV 文件中提供测试数据，另一种是更程序化的方式，您可以在 Java 中插入表格。在本文中，我们将研究基于文件的方法。

使测试配置单元脚本成为可能的库叫做[配置单元运行器](https://github.com/klarna/HiveRunner)。这是一个由 Klarna 构建并开源的库。它是一个零安装库，这意味着您只需导入一个 java 库并编写您的测试，不需要进一步集成任何东西。这意味着没有 Hadoop，HDFS，蜂巢安装是必要的。

假设我们创建配置单元模式的脚本在`main/resources/hive/create`下

正如您可能猜到的那样，测试需要通过执行这些脚本来准备，以获得初步的表格。然后，我们将拥有生成`main/resources/hive/report`下的报告的转换脚本

并且测试数据文件在`test/resources/StudentCountReportTest/successCase1`下；

**school.csv**

```
1,Cumhuriyet İlköğretim Okulu 2,Atatürk Lisesi 3,Samsun Anadolu Lisesi
```

**student.csv**

```
1,ali,yılmaz,21-09-1991,1 2,mehmet,yılmaz,22-07-1991,1 3,veli,kal,23-08-1990,1 4,şaban,yaşar,24-05-1990,1 5,ahmet,güngör,25-03-1990,1 6,seda,akyüz,26-07-1990,2 7,büşra,kilit,27-01-1990,2 8,ali,kıymık,28-06-1990,3 9,veysel,bolu,29-05-1990,3 10,ahmet,dal,21-09-1990,3
```

现在是时候编写我们的测试了；

我来解释一下代码；`hiveShell`默认自动启动。如果你想在启动前设置一些变量，你需要在注释中设置`atuoStart`为假，在`setUp`中设置你的变量，然后手动启动 hiveShell。我在这里不需要它。

*   首先，创建我们脚本中使用的数据库(`setUp`)。
*   执行创建脚本，并创建测试用例所需的所有表格(`setUp`)。
*   插入来自`school.csv`和`student.csv`文件(`testSuccessCase1`)的学校和学生测试数据
*   执行计算并填充`student_count_report`表的脚本(`testSuccessCase1`)。
*   检索`student_count_report`结果集并断言它(`testSuccessCase1`)。

在我的测试数据文件里，有 **10** 个学生。 **5 名**学生在**" cumhuriyet i̇lköğretim 奥库卢"**学校， **2 名**学生在 **"Atatürk Lisesi"** 学校， **3 名**学生在 **"Samsun Anadolu Lisesi"** 学校。你可以看到我的期望，你可以运行并看到这个测试将通过，希望:-)

您可以在[https://github . com/javrasya/blog-examples/tree/master/hive/unit test](https://github.com/javrasya/blog-examples/tree/master/hive/unittest)查看并使用本文中的示例