# 使用 MoviePy 在 Python 中创建自己的基本视频编辑软件

> 原文：<https://medium.com/analytics-vidhya/create-your-own-basic-video-editing-software-in-python-using-moviepy-fcb229153f5c?source=collection_archive---------2----------------------->

## 伟大的图书馆创建自己的视频编辑软件

![](img/b70ef00de0259994a605c2175a1411ff.png)

**目录**

1.  介绍
2.  装置
3.  你能做的事情
4.  从图片目录创建视频
5.  添加(或更改)视频的音频
6.  添加水印和简介
7.  结论

# **简介**

有时我们想编辑视频的基本任务，如添加介绍或结尾，连接不同的视频，添加水印等。为您的项目报告视频或任何个人项目。为此，你要么使用现有的在线工具，要么使用付费软件。为此，你需要有一些相关的经验，但是如果你知道如何用 python 编码，那么这个库会非常有用。

所以，最近我偶然发现了一个叫做 [MoviePy](https://zulko.github.io/moviepy/index.html) 的很棒的图书馆，它是最近发行的。所以，我想分享一下会很棒。

# **安装**

1.  Pip 方法:

```
(sudo) pip install moviepy
```

2.手动安装:

从这个[链接](https://github.com/Zulko/moviepy)下载 zip 文件。然后运行以下代码:

```
(sudo) python setup.py install
```

# **你可以用这个库做的事情:**

基本上，你可以在你的视频文件上执行所有基本的和一些高级的效果。这里有一些例子。

*   非线性视频编辑
*   从视频创建 gif
*   将音频添加到视频中
*   矢量动画
*   3D 动画
*   数据动画
*   连接不同的视频

还有很多。在接下来的部分中，我展示了一些您可以使用 MoviePy 完成的编码示例。你可以点击查看其他例子[。](https://zulko.github.io/moviepy/examples/examples.html)

# **从图片目录创建视频:**

当我们度假归来时，我们带着如此多的照片和视频，我们将这些时刻永久地保存下来。所以，我们能做的就是把那些图片做一个小视频，加上一些合适的音乐。所以，无论何时我们看它都会想起那个地方。

现在，如果你想这样做，那么你必须有任何沉重的视频编辑软件和一些技能来工作。但是有了这个库，你就可以不需要这些了。让我们跳到编码部分。

首先，进口我们需要的必需品。

```
from moviepy.editor import ImageSequenceClipfrom PIL import Image
```

现在，指定包含所有图像的输入目录和输出视频文件名。

```
thumbnail_dir = os.path.join(SAMPLE_OUTPUTS, "thumbnails")output_video = os.path.join(SAMPLE_OUTPUTS, 'final_div_to_vid.mp4')
```

这里，SAMPLE_OUTPUT 是我的输出文件夹的预定义路径。

现在，将目录中所有图像文件的路径添加到列表中。

```
this_dir = os.listdir(thumbnail_dir)filepaths = [os.path.join(thumbnail_dir, fname) for fname in this_dir if fname.endswith("jpg")]
```

因为我想让这些图片按照特定的顺序排列，所以我把它们重新命名为数字，这样我就可以更容易地用字典对它们进行排序。

```
directory = {}for root, dirs, files in os.walk(thumbnail_dir): for fname in files: filepath = os.path.join(root, fname) try: key = float(fname.replace(".jpg", "")) except: key = None if key != None: directory[key] = filepath new_path = []for k in sorted(directory.keys()): filepath = directory[k] new_path.append(filepath)
```

在这里，我创建了保存顺序和路径的字典。然后我根据关键字对它们进行了排序。现在，new_path 是按顺序包含所有图像路径的列表。

最后，我们将使用 ImageSequenceClip 通过指定 fps 或视频的持续时间来创建所有图像的视频。

```
clip = ImageSequenceClip(new_path, fps=5)
clip.write_videofile(output_video)
```

从图片创建视频的完整代码如下。

# **添加(或更改)视频的音频:**

有时，在录制视频时，我们有噪音，或者我们想在录制后给视频添加声音。然后我们可以使用这个库添加音频。

导入所需的模块。这里，[音频文件剪辑](https://zulko.github.io/moviepy/ref/AudioClip.html#audiofileclip)和[视频文件剪辑](https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#videofileclip)。

AudioFileClip 类提供了处理音频文件的方法，VideoFileClip 提供了处理视频文件的方法。

```
import os
from moviepy.editor import AudioFileClip, VideoFileClip
```

现在，用户需要输入:

*   视频路径
*   音频路径
*   最终视频输出目录
*   最终视频名称
*   音频文件的开始和结束时间

```
org_video_path = input("Enter the video path: ")audio_path = input("Enter the audio path: ")final_video_path = input("Enter the output folder path: ")final_video_name = input("Enter the final video name: ")start_dur = int(input("Enter the starting duration in seconds: "))end_dur = int(input("Enter the ending duration in seconds: "))final_video_path = os.path.join(final_video_path, final_video_name)
```

创建一个 VideoFileClip 类的对象。

```
video_clip = VideoFileClip(org_video_path)
```

如果您想使用视频的原始音频，您可以按如下方式提取:

```
original_audio = video_clip.audiooriginal_audio.write_audiofile(og_audio_path)
```

现在，创建音频文件的对象，然后选择用户指定的音频部分。

```
background_audio_clip = AudioFileClip(audio_path)bg_music = background_audio_clip.subclip(start_dur, end_dur)
```

最后，设置音频为视频。

```
final_clip = video_clip.set_audio(bg_music)final_clip.write_videofile(final_video_path, codec='libx264', audio_codec="aac")
```

执行该任务的完整代码如下:

# **添加水印和简介:**

有了这个库，您还可以通过使用 TextClip 类来添加水印和简介。让我们添加视频的基本介绍和水印。

```
import osfrom moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
```

现在，您已经熟悉了 VideoFileClip 和 AudioFileClip 类。所以，在这里我将简单解释一下 [TextClip](https://zulko.github.io/moviepy/ref/VideoClip/VideoClip.html#textclip) 和[compositevideocip](http://CompositeVideoClip)类。

TextClip 类用于通过指定参数来创建包含文本的视频剪辑。这个类提供了许多参数和方法来设置字体样式、字体大小、字体颜色、文本对齐、剪辑持续时间、位置、fps 等。

CompositeVideoClip 可用于连接两个不同的视频。它将视频列表作为参数。这个类也提供方法和许多参数和方法来设置音频，fps，持续时间等。

现在，在这里我要求用户插入视频路径，输出文件夹，最终视频名称，音频路径和水印，他们希望有视频。

```
org_video_path = input("Enter the video path: ")final_video_path = input("Enter the output folder path: ")final_video_name = input("Enter the final video name: ")audio_path = input("Enter the final video name: ")watermark = input("Enter the watermark: ")final_video_path = os.path.join(final_video_path, final_video_name)
```

现在，我们将创建视频和音频文件的对象，并根据视频持续时间修剪音频。

```
video_clip = VideoFileClip(org_video_path)audio_clip = AudioFileClip(audio_path)final_audio = audio_clip.subclip(25, 40) w, h = video_clip.sizefps = video_clip.fps
```

视频大小有助于创建介绍或水印等文本剪辑。

现在，让我们设计我们想要的介绍。

```
intro_duration = 5intro_text = TextClip("Hello world!", fontsize=70, color='white', size=video_clip.size)intro_text = intro_text.set_duration(intro_duration)intro_text = intro_text.set_fps(fps)intro_text = intro_text.set_pos("center")intro_music = audio_clip.subclip(25, 30)intro_text = intro_text.set_audio(intro_music)
```

在这里，我用文本、字体大小、颜色和剪辑大小创建了 TextClip 类的对象。之后，我设置了 5 秒钟的介绍时间，然后 fps 和视频的 fps 一样，然后是文本的位置。最后，我在介绍中加入了音频。

为了创建水印，我们几乎要做和上面一样的事情。

```
watermark_size = 50watermark_text = TextClip(watermark, fontsize=watermark_size, color='black', align='East', size=(w, watermark_size))watermark_text = watermark_text.set_fps(fps)watermark_text = watermark_text.set_duration(video_clip.reader.duration)watermark_text = watermark_text.margin(left=10, right=10, bottom=2, opacity=0)watermark_text = watermark_text.set_position(("bottom"))
```

所以，唯一的区别是我设置了边距、对齐方式并缩小了字符的大小。在这里，对齐方式要么是“东”对右，要么是“西”对左。

```
watermarked_clip = CompositeVideoClip([video_clip, watermark_text], size=video_clip.size)watermarked_clip = watermarked_clip.set_duration(video_clip.reader.duration)watermarked_clip = watermarked_clip.set_fps(fps)watermarked_clip = watermarked_clip.set_audio(final_audio)
```

在上面的部分，我结合了原始视频和水印剪辑，然后设置视频的持续时间，fps 再次与视频相同，并设置音频。

最后，我们连接视频并保存它们。

```
final_clip = concatenate_videoclips([intro_text, watermarked_clip])final_clip.write_videofile(final_video_path, codec='libx264', audio_codec="aac")
```

添加简介和水印的完整代码如下:

# **结论**

所以，在这里我只是展示了一个基本的例子，你可以不安装任何沉重的软件或使用任何在线工具。当然，你可以做的比我刚才用 MoviePy 展示的更多。

你可以在这里找到我的 Github 库[。](https://github.com/vyashemang/movie_editor)

不要犹豫，在下面的评论区说出你的想法。

感谢您阅读这篇文章。

此外，请查看我关于[推荐系统](https://hackernoon.com/popularity-based-song-recommendation-system-without-any-library-in-python-12a4fbfd825e?source=---------4------------------)和[部署机器学习模型](https://towardsdatascience.com/deploy-a-machine-learning-model-using-flask-da580f84e60c)的其他文章。

**参考文献**:

[电影播放](https://www.youtube.com/watch?v=m6chqKlhpPo)

[企业家编码](https://www.youtube.com/watch?v=m6chqKlhpPo)