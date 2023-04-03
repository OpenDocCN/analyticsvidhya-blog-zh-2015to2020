# 使用开放自然语言处理的 Java 命名实体识别

> 原文：<https://medium.com/analytics-vidhya/named-entity-recognition-in-java-using-open-nlp-4dc7cfc629b4?source=collection_archive---------6----------------------->

![](img/207ae9141c117bbf90bc5d2d03ecf50e.png)

毛罗·利马在 [Unsplash](https://unsplash.com/s/photos/text-highlight?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

# **简介**

您可能想知道，我们如何从文本中提取信息，或者我们如何仅使用 Java 从文本中提取关键字。当我们想到使用自然语言处理时，我们通常会在 python 中得到稳定的库，如 NLTK、spacy 等。因此，您通常会用 python 来实现这一点。当涉及到与可能用 java 编写的现有系统集成时，您可能最终会在 python 代码上编写 rest 包装器，以便系统可以与之通信。但是这样也有很多弊端。

1.  构建认证和授权机制，并确保 rest API 遵循安全协议
2.  为托管和部署此应用程序设置单独的环境
3.  故障点可能会增加

在我从事的一个项目中，我也遇到了类似的问题。我必须从维护电子邮件中提取数据中心的不同维护信息，如 ID、维护日期和提供商，并相应地创建票证。自动票据创建是用 Mulesoft 编写的，Mulesoft 是一个基于 spring 的 java 集成工具。

我可以实现一个基于正则表达式的方法，在那里我可以编写提取这些实体的模式。但是有许多不同的模式需要注意，这不是一个可行的解决方案。然后我知道这些信息已经存在于手工处理的现有票据中。因此，我想到建立一个 NER 模型，它可以通过用包含这些信息的门票数据集训练模型来提取这些实体。

我查看了许多提供培训定制 NER 模型选项的库，其中一个是非常受欢迎的生产级库空间。由于创建票证的集成已经出现在 Java 中，所以用 python 构建模型并将其与 rest wrapper 集成有点难以克服，因为上面列出了缺点。

因此，我遇到了一个由 Apache 命名为 Open NLP 的库。它有一个非常酷的 NER 模型，这是一个基于 java 的库，通过使用 Java 组件，它也可以很容易地用于 Mulesoft ESB。

让我们深入了解如何在 Open NLP 中创建一个定制的 NER 模型。

# **第一步:添加库**

下面是可以添加到 maven 项目的 pom.xml 中的依赖项，或者您也可以下载 jar 并将其添加到构建路径中。

```
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-tools</artifactId>
    <version>1.9.2</version>
</dependency>
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-uima</artifactId>
    <version>1.9.2</version>
</dependency>
```

# **步骤 2:准备训练数据集**

需要提供给开放 NLP 模型的数据集应该出现在一个文本文件中。该训练数据集应该包含文本，该文本包含需要提取的实体或关键字，并且这些实体应该用开始和结束标签进行注释。开始标签将表示需要提取的实体的开始。它应该遵循以下语法:<entity_type.entity_name>，其中 entity_name 是可选的，它可以在所有不同种类的将被注释的实体中保持通用。entity_name 是必需的，用于标记实体。<end>用来标记实体的结束。</end></entity_type.entity_name>

例如，在我的训练数据集中，训练数据集由不同电子邮件的正文组成，其中维护日期用开始和结束标签进行了注释。不同的电子邮件正文被放入文本文件的不同行中。

```
Dear Network User, Please be advised that the network will be unavailable from 01:00am to 05:30am on <START:maint.mdate> November 12th, 2014 <END> . This period of downtime will be scheduled for necessary updates to be applied to the network servers. We apologise for the inconvenience that this may cause. Kindly inform the IT Service Desk (at ext. 1234) of any concerns that you may have about the planned outage. Kind regards, abc name Network Administrator.Due to system maintenance, Certain account related features on Net Banking would not be available till <START:maint.mdate>Monday 6th September 17:00 hrs <END>. Credit Card Enquiry, Demat, and Debit Card details would continue to be available. We regret the inconvenience caused
```

准备训练数据集时，需要注意以下几点

1.  当注释标记前后有空格时，该模型可以工作。
2.  训练数据集应该包含每种文本至少 7-10 条记录，以便模型学习模式并正确预测
3.  训练数据集的每个记录应该出现在单独的行中(即由\n 换行符分隔)
4.  也可以应用标准的数据预处理技术，如删除标点符号、停用词、非 ascii 字符等

# **第三步:建立模型**

导入以下模块

```
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import opennlp.tools.namefind.BioCodec;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.NameSample;
import opennlp.tools.namefind.NameSampleDataStream;
import opennlp.tools.namefind.TokenNameFinder;
import opennlp.tools.namefind.TokenNameFinderFactory;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.util.InputStreamFactory;
import opennlp.tools.util.MarkableFileInputStreamFactory;
import opennlp.tools.util.ObjectStream;
import opennlp.tools.util.PlainTextByLineStream;
import opennlp.tools.util.Span;
import opennlp.tools.util.TrainingParameters;
```

读取步骤 2 中准备的训练数据集(training_dataset.txt)文本文件

```
InputStreamFactory in = null;
try {
    in = new MarkableFileInputStreamFactory(new File("AnnotatedSentences.txt"));
} catch (FileNotFoundException e2) {
    e2.printStackTrace();
}
ObjectStream sampleStream = null;
try {
    sampleStream = new NameSampleDataStream(
        new PlainTextByLineStream(in, StandardCharsets.UTF_8));
} catch (IOException e1) {
    e1.printStackTrace();
}
```

我们可以改变训练参数，例如迭代次数、损失截止参数和算法类型(目前开放的 NLP 仅支持 Maxent 和朴素贝叶斯)

```
// setting the parameters for training
TrainingParameters params = new TrainingParameters();
params.put(TrainingParameters.ITERATIONS_PARAM, 70);
params.put(TrainingParameters.CUTOFF_PARAM, 1);
paramaters.put(TrainingParameters.ALGORITHM_PARAM, 'MAXENT');
```

我们通过传入诸如编解码器、语言、训练数据集等参数来训练模型。训练之后，我们可以用一个名字将模型保存到文件系统中，这个名字将用于以后提取实体。

```
// training the model using TokenNameFinderModel class
TokenNameFinderModel nameFinderModel = null;
try {
nameFinderModel = NameFinderME.train("en", null, sampleStream,
params, TokenNameFinderFactory.create(null, null, Collections.emptyMap(), new BioCodec()));
} catch (IOException e) {
e.printStackTrace();
}
// saving the model to "ner-custom-model.bin" file
try {
File output = new File("ner-custom-model.bin");
FileOutputStream outputStream = new FileOutputStream(output);
nameFinderModel.serialize(outputStream);
} catch (FileNotFoundException e) {
e.printStackTrace();
} catch (IOException e) {
e.printStackTrace();
}
```

从[开放 NLP 官网](http://opennlp.sourceforge.net/models-1.5/en-token.bin)下载分词器模型，这个将用于对句子进行分词，因为模型需要以分词的形式出现的文本。我们加载模型，然后输入我们想要测试模型的标记化测试句子。

```
sentence = "<put in the sample sentence that you want to test here>"// Tokenise sentences
InputStream inputStreamTokenizer = new FileInputStream("en-token.bin");
TokenizerModel tokenModel = new TokenizerModel(inputStreamTokenizer);
TokenizerME tokenizer = new TokenizerME(tokenModel);
tokens =  tokenizer.tokenize(sentence);//Load the model created above
InputStream inputStream = new FileInputStream("ner-custom-model.bin");
TokenNameFinderModel model = null;
try {
    model = new TokenNameFinderModel(inputStream);
} catch (IOException e) {
    // TODO Auto-generated catch block
    e.printStackTrace();
} 
NameFinderME nameFinder = new NameFinderME(model);
Span nameSpans[] = nameFinder.find(tokens);// testing the model and printing the types it found in the input sentence
for(Span name:nameSpans){
String entity="";
System.out.println(name);
for(int i=name.getStart();i<name.getEnd();i++){
    entity+=tokens[i]+" ";
}
System.out.println(name.getType()+" : "+entity+"\t [probability="+name.getProb()+"]");
```

# 为什么不是斯坦福核心 NLP？

你可能想知道，为什么我不选择斯坦福核心 NLP，它也是一个广泛流行的 NLP Java 库。原因是 Stanford core NLP 需要大量的训练数据集来训练客户 NER 模型，而不像 Open NLP 可以用很少的训练数据集来学习模式。也许在我的下一篇博客中，我会尝试解释如何用斯坦福核心自然语言处理训练一个定制的 NER 模型，并与开放自然语言处理进行比较。

我希望你喜欢我在 Open NLP 上的帖子，如果是的话，请与你感兴趣的朋友和同事分享。还有，多给这个帖子鼓掌，会鼓励我多写帖子。如果你有任何疑问，需要帮助来完成这段代码，请在下面的评论区留下你的想法。

直到下一个帖子，谢谢！！！