# 基于 Apache Open NLP 的文本分类

> 原文：<https://medium.com/analytics-vidhya/text-classification-based-on-apache-open-nlp-317f82f4b1b0?source=collection_archive---------5----------------------->

## Apache Open NLP 产品中印第安人的力量和智慧允许您使用机器学习方法解决准备和分类文本的复杂任务。

![](img/14a2da422ba41b1c23ff5e491cb238cd.png)

www.freepik.com

# *问题陈述*

在过去的几年里，机器学习为金融行业和智能经济开辟了新的机遇。例如，对营销和金融行业中的产品进行自动分类的任务就非常相关。为了自动生成产品目录而定义类别是一项流行的任务。另一项任务是确定买家感兴趣的对象，准备主要的消费篮子。这些和其他领域需要数据处理自动化。

Apache Open NLP 库提供了各种工具来自动化这些任务，包括分类器的实现。分类中最有力的方法之一是最大熵准则。我们将不深究数学计算和概率论。你可以在这里看到(链接)。让我们开始检查它是如何工作的。首先，让我们为我们的任务导入 java 库。

# *库的使用及其导入*

```
**package** nlp.textclass;

**import** java.io.FileInputStream;
**import** java.io.InputStream;

**import** opennlp.tools.postag.*;
**import** opennlp.tools.tokenize.*;
**import** opennlp.tools.util.*;
**import** java.io.*;
**import** java.util.*; 
```

为了进一步的工作，我们需要一个文本处理库和数据输入/输出工具。出于这些目的，我们将采用 Apache 基金会的开放 NLP 产品和用于组织输入/输出流的 [*java.util*](https://mvnrepository.com/artifact/com.cedarsoftware/java-util/1.8.0) 库。我们开始准备数据吧。

# *培训数据准备*

在直接进行自动分类模型的生成之前，我想接收用于处理的数据并适当地准备它们。准备好数据和标签后，看起来是这样的。

这里第一个字是数据，第二个字是标签。

```
String[] product = [“cars_cargo” ,”ipad_electronic” ,”game_entertainment” ,”iphone_electronic” ,”stereo_electronic” ,”mouse_electronic” ,”keyboard_electronic” ,”tablet_electronic” ,”technic_electronic” ,”electronic_electronic” ,”analise_medical” ,”drugs_medical” ,”gas_oil” ,”shave_goods” ,”trousers_clothes” ,”bottle_alcohol” ,”bycicle_bike” ,”bike_bike” ,”stuff_clothes” ,”vine_alcohol” ,”water_goods” ,”jeance_clothes” ,”food_food”];
```

我们得到文件 **gen_data.txt.** 训练数据可能包括**所有可用**的 80%。如果我们想解决一个真正的问题，我们需要更多的数据！

![](img/a26cf0d57a57f8e5b85339b3859065d7.png)

[www.freepik.com](http://www.freepik.com)

它必须是**5000 以上**或【50000 以上标明的要素。

当我们得到规范化和结构化的数据时，我们就可以开始构建模型了。该模型由一组训练数据生成。

# 模型生成

在训练并获得**模型-Maxent 之后，**我们可以在各种测试数据上对其进行测试。**测试数据可以是所有数据的 20%。但是在测试模型之前，我们将为输入文本添加处理程序。这是一个句子检测器和分词器。**

# *句子检测器和分词器*

要对产品进行分类，必须在输入文本中选择单独的单词。在这个例子中，我们将文本分成单独的句子，然后分成单词。我们建立我们的**词性模型(POSmodel)** ，之后，than。

```
POSModel model = POSTaggerME.*train*("en", sampleStream, TrainingParameters.*defaultParams*(), posModel.getFactory());
Sequence sequences[] = posTagger.topKSequences(message);
```

训练结果是这样的。

```
POS model started
Indexing events using cutoff of 5Computing event counts… done. 374 events
 Indexing… done.
Sorting and merging events… done. Reduced 374 events to 366.
Done indexing.
Incorporating indexed data for training… 
done.
 Number of Event Tokens: 366
 Number of Outcomes: 47
 Number of Predicates: 99
…done.
Computing model parameters …
Performing 100 iterations.
 1: … loglikelihood=-1439.955203039556 0.016042780748663103
 2: … loglikelihood=-1153.0290547162213 0.27807486631016043
 3: … loglikelihood=-1027.8934671969257 0.2914438502673797
...99: … loglikelihood=-328.84956863946474 0.7593582887700535
100: … loglikelihood=-327.86155121782036 0.7593582887700535
Model generated…
Process finished with exit code 0
```

如果模型已经存在，我们可以从文件中加载它。

```
InputStream inPosStream = getClass().getClassLoader().getResourceAsStream("en-pos.dat");
POSModel posModel = new POSModel(inPosStream);
inPosStream.close();
POSTaggerME posTagger = new POSTaggerME(posModel);
```

做得好，现在让我们继续测试，这在代码中有记录。然后每个单词都会被分类和标注(TAG)，例如:

```
-> OpenNLP: tag detector
boots it is clothes
dress it is clothes
```

我们得到一个模型，然后在文本上运行它，文本被分成单词。它检测单词并使用 Java 代码对它们进行分类。我们使用了 OpenNLP 库并创建了 NLPClassifier 类(详细代码在 [Github](https://github.com/AlexTitovWork/NLPclassifier/blob/master/src/main/java/NLPClassifier.java) 上)

```
public class NLPClassifier {…}
```

让我们测试一下**句子检测器**。用作**分割器**的**标记器**。**后置分类器**，在主函数中作为**分类器**使用。你可以在 [Github](https://github.com/AlexTitovWork/NLPclassifier/blob/master/src/main/java/NlpProductClassifier.java) 上找到代码

POS 标记器的完整代码和经过测试的**二进制** **模型**都在资源库中。

[](https://github.com/AlexTitovWork/NLPclassifier) [## AlexTitovWork/NLP 分类器

### MLP 分类器它是基于 OpenNLP 库的产品分类器。使用 OpenNLP 库的营销数据分类器 OpenNLP…

github.com](https://github.com/AlexTitovWork/NLPclassifier) 

其他对象

[](https://github.com/AlexTitovWork) [## AlexTitovWork -概述

### 在 GitHub 上注册您自己的个人资料，这是托管代码、管理项目和与 40…

github.com](https://github.com/AlexTitovWork) 

祝福你，亚历克斯！