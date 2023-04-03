# 使用 BERT 进行长文章的问答

> 原文：<https://medium.com/analytics-vidhya/question-and-answer-for-long-passages-using-bert-dfc4fe08f17f?source=collection_archive---------15----------------------->

![](img/9385fe0f19d9a6c5ec98b98a15771485.png)

[https://d 827 xgdhgqbnd . cloudfront . net/WP-content/uploads/2019/04/09110726/Bert-head . png](https://d827xgdhgqbnd.cloudfront.net/wp-content/uploads/2019/04/09110726/Bert-Head.png)

BERT，即来自变压器的双向编码器表示，是一种预训练语言表示的新方法，它在各种自然语言处理(NLP)任务上获得了最先进的结果。可以肯定地说，它正在席卷 NLP 世界。BERT 是由谷歌开发的，Nvidia 已经创建了一个使用 TensorRT 的优化版本。(【https://github.com/google-research/bert】T2 和[https://devblogs.nvidia.com/nlu-with-tensorrt-bert/](https://devblogs.nvidia.com/nlu-with-tensorrt-bert/))

BERT 的一个缺点是在执行问答时只能查询短文。当文章达到一定长度后，就找不到正确答案了。要运行问答查询，您必须提供要查询的文章以及您试图从文章中回答的问题。

我已经创建了一个脚本，允许您查询较长的段落并获得正确的答案。我取一个输入段落，并将其分成由\n 分隔的段落。然后查询每个段落以试图找到答案。所有返回的答案都放在一个列表中。然后对列表进行分析，找出概率最高的答案。这将作为最终答案返回。当您运行该脚本时，您将需要更改路径以符合您的设置。脚本可以在 https://github.com/pacejohn/BERT-Long-Passages 的 GitHub 上找到。

# 设置

为了正确运行脚本，您需要确保创建了 Docker 容器。在运行查询之前，请确保启动 TensorRT 引擎。以下是英伟达说要做的和我正在做的步骤。

从主目录中，运行以下命令。这需要一段时间。

克隆 TensorRT 存储库并导航到 BERT 演示目录
*git 克隆—递归*[*https://github.com/NVIDIA/TensorRT*](https://github.com/NVIDIA/TensorRT)*&&CD TensorRT/demo/BERT*

创建并启动 docker 映像
*sh python/Create _ docker _ container . sh*

构建插件并下载微调后的模型
*CD TensorRT/demo/BERT&&sh python/Build _ examples . sh base fp16 384*

构建 TensorRT 运行时引擎并启动它。如果你不做这件事，你就不能做其他任何事。
*nohup python python/Bert _ builder . py-m/workspace/models/fine-tuned/Bert _ TF _ v2 _ base _ fp16 _ 384 _ v2/model . ckpt-8144-o Bert _ base _ 384 . engine-B1-s 384-c/workspace/models/fine-tuned/Bert _ TF _ v2 _ base _ fp16 _ 384 _ v2>tensorrt . out&*

启动引擎后，您可以在脚本
*python/Bert _ inference _ loop . py*中运行问答查询

一个警告是，TensorRT 引擎将在一段时间后终止。在执行查询之前，请确保它正在运行。

# 我用于查询的文件

我可以用 3 个文件作为输入段落。请随意用你自己的文章来试一试。

22532_Full_Document.txt —这是我使用的完整文档。如果你问一个关于第一部分的问题，它会返回正确的答案。如果你问一个关于后面部分的问题，它不会找到答案。

22532 _ Short _ Document _ With _ answers . txt—这是一个简短的段落，包含了对查询的回答。如果你使用和我一样的查询，它会找到 2 个答案。概率较大的是正确答案。

22532 _ Short _ Document _ Without _ answers . txt—这是一段简短的短文，其中没有问题的答案。如果你使用和我一样的查询，它将找不到任何答案。

提出的问题是*“有多少患者在 12 岁时经历了复发？”*随意实验。

欢迎大家的反馈和建议。请务必在 Twitter @pacejohn 上关注我，并查看我在 https://www.ironmanjohn.com/的博客。