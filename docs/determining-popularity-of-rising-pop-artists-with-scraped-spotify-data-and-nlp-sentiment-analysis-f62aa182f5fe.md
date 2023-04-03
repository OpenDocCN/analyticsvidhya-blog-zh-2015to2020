# 利用 Spotify 数据和 NLP 情感分析确定新兴流行歌手的受欢迎程度

> 原文：<https://medium.com/analytics-vidhya/determining-popularity-of-rising-pop-artists-with-scraped-spotify-data-and-nlp-sentiment-analysis-f62aa182f5fe?source=collection_archive---------12----------------------->

![](img/1b25e3c25b3832e9f6938e93306a2407.png)

执行摘要(图片由作者 James Pecore 制作)

## 问题陈述:

Spotify 使用其流行度参数来对歌曲、专辑和艺术家进行排名。这个“流行度”指标是基于用户从 Spotify 上观看歌曲的频率。但这一指标仅显示了最近艺术家的总体受欢迎程度(不是根据流派的受欢迎程度，也不是根据歌曲/歌词内容的受欢迎程度)。因此，历史上非常流行的经典歌曲被忽略了。此外，在他们的流派中非常受欢迎的艺术家变得被忽视，因为来自更受欢迎的流派如“流行”的较高权重的艺术家我们需要一个新的受欢迎程度的衡量标准。事实上，我们需要的不止一个。

以下问题将帮助我们在更大的数据背景下重新评估 Spotify 的流流行度指标:

1.根据音乐本身的一些方面，我们可以对一首歌的受欢迎程度说些什么:比如可舞性、活力和声音？

2.根据艺术家歌词的内容——诗歌的语言内涵和氛围，我们可以对一首歌的受欢迎程度说些什么？

3.这些因素是如何影响我们预测一位艺术家或一首歌的受欢迎程度的？

4.最后，当使用回归建模、分类建模和 NLP 聚类来预测音乐艺术家的受欢迎程度时，如何评估是否信任 Spotify 的受欢迎程度排名？

# 执行摘要:

![](img/1b25e3c25b3832e9f6938e93306a2407.png)

我用 APIs Spotipy 和 Genius 创建了两个不同的数据集。我还使用了 Zaheen Hamidani 的 Kaggle 数据集来增加我的数据量。

接下来，我为大约 150，000 首歌曲的数据集建立了各种各样的回归模型。这些模型试图根据歌曲的音乐属性(如能量、效价、形态、拍号和其他特征)准确预测歌曲的“流流行度”。我还使用许多不同的分类模型来衡量我们是否可以根据这些相同的歌曲属性来预测一首歌曲是否受欢迎(在 0 到 100 的范围内超过 75%的受欢迎程度)。

对于歌词属性，我使用 Spotify 播放列表中较短的歌曲列表(只有 700 首歌曲)作为收集歌词的基础。我从 Genius 的歌词库中抓取了每首歌的歌词。我使用情感分析和 NLP(计数矢量器)对每首歌最常见的词/情感进行 EDA。最后，我试着用它的流行程度来评价大部分常用词和歌曲情绪是否有相关性。

# **解释性数据分析**

![](img/c96cc1942c2e2599c484553a8b5d460a.png)

150，000 首 Spotify 歌曲的受欢迎程度分布，图片由作者 James Pecore 制作

![](img/a815794fbf37ddee84eff974e770e4bd.png)

歌曲属性与流流行度的相关性，由作者 James Pecore 制作的图像

作为数据科学家，我们应该感到惊讶的是，人们可以使用“响度”来如此准确地预测 Spotify 歌曲的“流流行度”。这是为什么呢？

嗯，“流流行”倾向于更喜欢最近制作的音乐(因为当前音乐更频繁地被流传输，因此比旧音乐更“流流行”)。

![](img/3fc535c08a6927998021fbc2da934221.png)

图片来自音乐技术学生(Itsaam)，引用作品中提供了链接

由于音乐压缩的历史，当代音乐(2007 年及以后)在流式播放时声音更大。因为 2000 年代后期的数字音乐创新允许音乐不那么压缩，所以数字形式的现代音乐仅仅被认为比早些年的数字化压缩更响亮。

我的观点是——响度不会让你的音乐在某一点上更受欢迎。如果是的话，“重金属”将会是我们最喜欢的音乐类型。

然而，声音似乎确实会影响一首歌的受欢迎程度。正如下面的信息图所详述的，更流行的歌曲通常较少使用原声音乐，而更多使用数字音乐。鉴于最近流行音乐在 Logic、Pro Tools、FL Studio 和 Ableton 等 Daw 中变得更加数字化的趋势，这些数据是有意义的。

![](img/3fcf2e0b4edfc145d09d2f52a54653b8.png)

歌曲属性(声音)与流流行度的相关性，由作者 James Pecore 制作的图像

# 回归建模:

![](img/470f84e26bed20af415300a01f75b4e6.png)

# 抒情分析:

情感分析是创建二进制单词的过程，以确定文本主体是否更接近一个极点或另一个极点。例如，我创建了一个“爱情”相关词和“心碎”相关词的二元结构。然后，我使用 CountVectorizer 对每首歌歌词中的每个单词进行矢量化。这将单词转换成数字向量，然后可以根据单词的相似性进行聚类。

最后，我创建了一个标准，将歌曲歌词的情感分析标准化为“爱情”歌曲接近+1 或“心碎”歌曲接近-1。然后，我可以使用这个抒情度量(以及其他情感分析二进制文件)作为建模的一个特征。

# 分类建模:

![](img/4b258f16b20245661e9dd47bf9287504.png)

# 聚类分析:

![](img/3847e6d74e2dca5f8112a8ea9071ed01.png)

# 推荐

对歌曲作者的一般建议:

*   将精力和舞蹈能力提高到平均值(60%)左右
*   降低声音强度，使用数字乐器/音乐制作
*   仅增加音量，以便于在手机上收听
*   如果你在歌里多提“爱”，也没什么坏处

推荐 1:空前的流人气

*   基于以下内容创建新的流行度指标:
*   "所有时间的流总数"
*   这将让我们对老歌和新歌进行分级
*   我们可以比较音乐的历史趋势和当前趋势，而不用担心流流行的不适当的缩放

建议 2:个人声望

*   为每个用户的歌曲带回一个 5 星或“一到十”的审查系统
*   这将让我们评估每个用户喜欢什么样的风格
*   这将允许我们为用户的最高评级歌曲创建回归模型和推荐系统，提高用户投票率

推荐 3:歌曲特色点评

*   为 Spotify 中的每首歌曲创建一个可选的功能评论部分
*   向量化特征检查中使用的词语
*   用这些向量创建情感分析
*   用这些矢量化的情感创建一个推荐系统

建议 4:个人研究

*   具有音乐教育背景的艺术家，如查理·普斯、利佐和 Lady Gaga，都拥有伯克利、MSM、NYU 和休斯顿大学等知名音乐大学的音乐学位
*   在你把艺术家的范围缩小到“前五名”之后，应该在某一点上对该提拔谁进行单独研究

# 进一步研究和未来项目

1.  使用并行编程(AWS)而不是串行编程(Jupyter)
    -处理所有 150，000 首歌词
    -扩展 NLP 对所有 150，000 首歌词执行情感分析
    -对所有 150，000 首歌词执行具有空间的 NLP 聚类
2.  利用流行歌曲的舆论进行情绪分析
    ——抓取新闻/Twitter/Reddit/Tumblr/等。所有歌曲的帖子
    -使用 NLP 来确定公众对艺术家的意见是-、0 还是+
3.  使用歌曲属性和评论创建推荐系统
    -在线发布或提交给唱片公司/流媒体公司

# 引用的作品

*   大会数据科学沉浸式 2020
*   量化音乐和音频。纽约市数据科学院，2019 年 6 月 3 日，NYC Data Science . com/blog/student-works/web-scrapeing/Spotify-x-billboard/。
*   乔治亚娃，埃琳娜，讲师。" HitPredict:使用 Spotify 数据预测广告牌点击率."2020 年的 ICML，由斯坦福大学的 Nicholas Burton 和 Marcella Suta 研究，2020 年 7 月 18 日。
*   阿什利·金格里斯基。" Spotify Web API:如何使用 Python 提取和清理热门歌曲数据."Ashley Gingeleski，2019 年 11 月 11 日，Ashley Gingeleski . com/2019/11/11/Spotify-we b-API-how-to-pull-and-clean-top-song-data-using-python/。
*   哈米达尼，扎辛。Kaggle，2019，[www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db](http://www.kaggle.com/zaheenhamidani/ultimate-spotify-tracks-db)。
    - Itsaam。"压缩的历史和发展."音乐技术学生，2013 年 8 月 11 日，music Tech Student . co . uk/music-technology/压缩的历史和发展/
*   战利品，稀有。"通过 Python 提取你最喜欢的艺术家的 Spotify 数据."Medium，2018 年 12 月 30 日，medium . com/@ rare loot/extracting-Spotify-data-on-your-favorite-artist-via-python-d 58 BC 92 a 4330。
*   帕西，雅各布。“Spotify 如何影响哪些歌曲流行(或不流行)。”MarketWatch，2018 年 6 月 18 日，[www . market watch . com/story/how-Spotify-influences-what-songs-be-popular-or-not-2018-06-18](http://www.marketwatch.com/story/how-spotify-influences-what-songs-become-popular-or-not-2018-06-18)。
*   皮埃尔，萨德拉赫。"使用 Python 分析 Spotify 前 50 首歌曲."Medium，Towards Data Science，2019 年 12 月 27 日，Towards Data Science . com/analysis-of-top-50-Spotify-songs-using-python-5a 278 dee 980 c。
*   萨胡，安普拉蒂姆。"使用 Spotify 的 API 和 Python 中的 Seaborn 对音乐品味进行国别视觉分析."中，走向数据科学，2020 年 6 月 12 日，走向 sdata Science . com/country-wise-visual-analysis-of-music-taste-using-Spotify-API-seaborn-in-python-77 F5 b 749 b 421。
*   斯帕西。spacy.io/. 2020 年 9 月。
*   面向开发者的 Spotify。developer.spotify.com/dashboard/. Spotify，2020 年 9 月。
*   阅读文件。2020 年 9 月，spotipy.readthedocs.io/en/2.16.0/.斯波蒂皮。
*   十二页文档。docs.tweepy.org/en/latest/. tweepy，2020 年 9 月。