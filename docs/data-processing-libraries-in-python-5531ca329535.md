# Python 中的数据处理库

> 原文：<https://medium.com/analytics-vidhya/data-processing-libraries-in-python-5531ca329535?source=collection_archive---------16----------------------->

## 为数据类型使用正确的库

![](img/28563a9bc9aaeccdde433b700c456c68.png)

随着大多数流行的库和大多数前沿技术的实现，Python 通常被推荐为机器学习相关项目的好选择。然而，当查看现有的所有不同的库时，可能会令人望而生畏，并且很难选择一对来开始。

机器学习建立在数据处理的基础上，模型的性能将在很大程度上依赖于您以合适的格式读取和转换数据以完成您希望完成的任务的能力。在本文中，我们将介绍不同的库，以及它们可以处理的数据类型。

# 表列数据

表格数据就是我们有时所说的*大数据、*，因为它以松散组织的行的形式出现，对应于样本和列，对应于特性(至少大多数时候是这样)。
[熊猫](https://pandas.pydata.org/)就是可以处理这类数据的库。它是为需要对序列(如移动平均线)进行大量操作的股票市场而创建的，已经发展成为一个功能齐全的库，可以很好地处理表格数据。

这里有一个例子。

```
import pandas as pd
df = pd.DataFrame('data.csv')
df.head()               # Prints head of the table
df.describe()           # Describes each column with common statistics
df.prices.plot()        # Will plot column "prices"
```

如你所见，语法很容易理解。有时难以理解的部分是`pandas`管理索引和选择的方式。它归结为您定义的索引列。如果没有参数传递给`pd.DataFrame`, pandas 会创建一个索引列，并在每一行递增。您可以使用不同的方法来选择一系列值。

# 文本数据

首先要注意的是，Python 自带了很多强大的内置文本处理功能。

```
raw_data = "Some text"
processed_data = raw_data.lower()
processed_data = raw_data.strip()
```

然而，自然语言处理涉及许多处理技术，如标记化、词条化，这些都可以使用 [NLTK](https://www.nltk.org/) 来实现。

```
import nltk
nltk.download()
tokens = nltk.tokenize(s)
```

对于更高级的自然语言处理，如果你的目标是优化管道，spacy 是一个可靠的选择。

# 音频和音乐数据

音频处理通过用于音频处理的 [librosa](https://librosa.github.io/librosa/) 和 [essentia](https://essentia.upf.edu/) 等库启用。例如，这些是在音乐信息检索研究社区中非常受欢迎的库。

对于象征性的音乐，例如当使用 MIDI 时， [mido](https://mido.readthedocs.io/en/latest/) 和 [pretty_midi](http://craffel.github.io/pretty-midi/) 是个不错的选择。更高级的是 [music21](https://web.mit.edu/music21/) ，这是一个强大的库，主要针对音乐学分析，具有广泛的抽象，例如将乐谱分为流、声部、轨道和测量对象。它也有一个简单的语法。

```
from music21 import converter
from music21 import note, chord
midi = converter.parse('file.mid')
# Lets print out notes and chords of the MIDI file
for element in midi.flat:
    if isinstance(element, note.Note):
        print("We have the note {} of duration {} at offset time {}".format(
            element.pitch,
            element.quarterLength,
            element.offset
        ) )
    elif isinstance(element, chord.Chord):
        print("We have the chord {} of duration {} at offset time {}".format(
            element.name,
            element.quarterLength,
            element.offset
        ) )
# Music21 has some nice display functions
midi.show("text")               # Print out all MIDI file with indentation reflecting hierarchy
midi.measures(5,10).show("midi")               # Show a PNG image of the score from measures 5 to 10
midi.measures(5,10).plot()                     # Display pianoroll of measures 5 to 10
```

# 形象

[Pillow](https://python-pillow.org/) 是 Python 中处理图像的库。它可以做图像编辑程序会做的事情。

```
from PIL import Image, ImageFilter
im = Image.open('/home/adam/Pictures/test.png')
new_im = im.rotate(90).filter(ImageFilter.GaussianBlur())
```

[scikit-image](https://scikit-image.org/) 也用于图像处理，并提供大多数可用的过滤器和算法。Opencv 是一个针对计算机视觉的库，可以用于处理视频或处理来自相机的数据。
如果您使用不常见或非常特殊的图像格式， [imageio](https://imageio.github.io/) 将能够向您的 python 脚本提供图像数据，这得益于其[广泛的支持格式](https://imageio.readthedocs.io/en/stable/formats.html)。

# 数据

以上所有库都有能力读取特定的数据格式。当这被转换成 python 对象和数据结构时， [numpy](https://numpy.org/) 通常会发挥作用，来操作这些数值。

在拿出深度学习火箭筒之前，建议使用 [sklearn](https://scikit-learn.org/stable/index.html) 、 [scipy](https://www.scipy.org) 和/或 [seaborn](https://seaborn.pydata.org/) 对数据进行一些分析。

# 结论

一旦执行了所有这些数据处理和分析，我们就可以获得足够的数据信息来考虑为我们的任务选择什么模型，并且希望机器学习技术可以充分发挥其潜力。

就我们想要处理的数据类型而言，本文涵盖了最常用的库。这些库是机器学习课程中常用的库，没有什么比在实际问题上使用这些工具积累一些经验更好的了！

阅读我网站上的文章。