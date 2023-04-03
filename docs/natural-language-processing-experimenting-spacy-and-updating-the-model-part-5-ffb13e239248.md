# 自然语言处理:空间实验与模型更新(五)

> 原文：<https://medium.com/analytics-vidhya/natural-language-processing-experimenting-spacy-and-updating-the-model-part-5-ffb13e239248?source=collection_archive---------2----------------------->

![](img/7d0870e86889e50a10457ce9473d77a7.png)

# 介绍

在前 4 篇文章中，我们已经说明了 Google 和 AWS NLP APIs 的用法。我们还试验了 spacy 库，从不同的文档中提取实体和名词。我们已经展示了如何使用 spaCy 的模式匹配函数来改进该模型。

我们想比以前的文章更进一步，试验 spacy 的高级功能。spaCy 允许更新统计模型并用新实体训练模型，而不使用“硬编码”匹配规则。这将允许根据我们的特定领域对模型进行微调。这对于实体识别或文本分类非常有用。

我们选择使用个人简介和职位描述，因为这是一个常见的用例，任何人都很容易理解，尽管我们认为使用这种算法来匹配候选人和职位是远远不够的。

# 使用预先训练的模型

如前一篇文章所示，让我们用通用职位描述文档上预先训练好的大型英语模型来解析文档，并让我们看看哪些实体已被识别:

```
import spacy spacy.load('en_core_web_lg') 
docJob=nlp(textCV) 
for ent in docJob.ents: 
   # Print the entity text and its label 
   if ent.label_=='PRODUCT': 
   print(ent.text, ent.label_,) 
SLQ ORG 
7 years DATE 
Statistics ORG 
Mathematics ORG 
C++ LANGUAGE 
Java LOC 
SLQ ORG 
S3 PRODUCT 
Spark ORG 
DigitalOcean ORG 
3rd ORDINAL 
Google Analytics ORG 
Site Catalyst, ORG 
Coremetrics ORG 
Adwords ORG 
Crimson Hexagon ORG 
Facebook Insights ORG 
Hadoop NORP 
Hive ORG 
Gurobi GPE 
MySQL GPE 
Business Objects ORG 
Glassdoor ORG
```

有趣的是，C++被认为是一种语言，而 Java 被认为是一个地缘政治实体，因为它是印度尼西亚的一个岛屿。例如 JavaScript 根本没有识别。这显然取决于模型被训练的背景类型，并且可能不特别是在计算机科学领域。

之前我们已经展示了我们可以将自己的规则添加到模型中。因此，让我们将我们自己的检测规则添加到模型中，以更接近我们想要的，即识别人们档案中的技术技能:

```
# define patterns we want to recognize 
patterns = [{"label": "PROG", "pattern": [{"lower": "java"}]}, {"label": "PROG", "pattern": [{"lower": "javascript"}]}] 
# define an entity ruler using predefined patterns 
ruler = EntityRuler(nlp, patterns=patterns,overwrite_ents=True) 
# add the ruler to the nlp pipeline 
nlp.add_pipe(ruler) 
# apply to the job document 
docJob=nlp(textJob) 
for ents in docJob.ents: 
# Print the entity text and its label 
  if ents.label_=='PROG': 
     print(ents.text, ents.label_,) 
Java PROG 
JavaScript PROG
```

现在，我们已经将 Java 和 JavaScript 确定为编程语言。

因此，我们能够将基于规则的实体识别添加到统计模型中。这允许以快速的方式对模型进行微调，以适应特定的领域和需求。

这很有趣，因为在这种情况下，模型将 Java 检测为一个地缘政治实体，将 JavaScript 检测为一个组织。这是自动完成的，不需要为这个特定的文档编写特定的规则。

因此，总之，我们可以说，不仅实体识别依赖于所使用的统计模型，而且显然依赖于用于训练的文档所涉及的领域类型。

# 培训和更新模型

spaCy 允许我们训练底层神经网络，并用我们特定的领域知识更新它。这是一个 coll 特性，因为这正是我们想要做的。首先让我们添加一些我们想要在代表性句子中检测的实体的例子:

```
trainData=[('Java is a programming language', {'entities': [(0, 4,'PROG')]}), ('I have 5 years experience in JavaScript', {'entities': [(27, 37,'PROG')]}), ('Extensive Java experience required', {'entities': [(10, 14,'PROG')]}), ('JavaScript is a programming language used mainly in front-end development', {'entities': [(0, 10, 'PROG')]}), ('Java is an object oriented programming language', {'entities': [(0, 4, 'PROG')]}), ('I have a long experience in project management', {'entities': []})]
```

它还建议举出反面例子，例如没有任何实体的句子。

现在，让我们使用这些新实体和训练示例来训练和更新模型:

```
# initialize a blank spacy model 
nlp = spacy.blank('en') 
# Create blank entity recognizer and add it to the pipeline 
ner = nlp.create_pipe('ner') 
nlp.add_pipe(ner) 
# Add a new label for programming language 
ner.add_label('PROG') 
# Start the training 
nlp.begin_training() 
# Train for 10 iterations 
for itn in range(10): 
  random.shuffle(trainData) 
  # Divide examples into batches 
  for batch in spacy.util.minibatch(trainData, size=2): 
    texts = [text for text, annotation in batch] 
    annotations = [annotation for text, annotation in batch] 
    # Update the model 
    nlp.update(texts, annotations)
```

现在，我们准备像以前一样在同一个文档上测试我们的模型。

```
docJob=nlp(textJob) 
for ents in docJob.ents: 
  # Print the document text and entitites 
  if ents.label_=='PROG': 
  print(ents.text, ents.label_,) 
Data PROG 
Job PROG 
Description PROG 
Job PROG 
Java PROG 
JavaScript PROG 
Site PROG 
Map PROG
```

这很酷，现在 Java 和 JavaScript 已经被 spaCy 神经网络识别了。不幸的是，它也将一些新词归类为 PROG，但这是一个好的开始。

*原载于 2019 年 5 月 26 日*[*https://smart lake . ch*](https://smartlake.ch/natural-language-processing-experimenting-spacy-and-updating-the-model-part-5/)*。*