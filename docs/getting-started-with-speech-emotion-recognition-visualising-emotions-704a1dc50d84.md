# 语音情感识别入门|可视化情感

> 原文：<https://medium.com/analytics-vidhya/getting-started-with-speech-emotion-recognition-visualising-emotions-704a1dc50d84?source=collection_archive---------5----------------------->

![](img/91d1c0b4713d24b1287d0b4e92ff20fb.png)

图片(作者):各种情绪的样本波形图及其相应的 MFCC 特征

随着当今世界人机交互系统和支持语音的应用的增加，我们的智能系统学习识别不同的情绪以进行更有效的交流已经变得不可或缺。越来越需要不仅从语音中提取语言信息，而且还要结合与之相关的情感。

语音情感识别(SER)是一项非常具有挑战性的任务，因为生成合适的训练数据需要大量的投资，并且注释具有很高的主观性。对于各种工业应用，有时甚至无法人工生成训练数据，因为没有训练有素的参与者。此外，高冗余度的注释是必须的，因为相同的信息可能会被不同的人根据情绪做出不同的解释。因此，这项技术并没有真正显示出很大的前景。

早期，工作更多地集中在基于启发式的方法来解决 SER，但现在这些方法正朝着使用人工神经网络以及注意机制和使用频谱图、MFCCs 和其他声学特征的更复杂的特征融合的方向发展。

总的来说，我对机器学习问题的可视化感到兴奋，因为我开始使用 Tensorboard 来可视化各种领域的特征。在处理原始音频和音频特征时，可视化通常对于更好地理解数据变得至关重要。听音频片段很耗时，可能需要多次重复才能客观地定义音频中的内容，但可视化有助于快速构建上下文。

# 感受情绪是怎样的

这个博客的目的是展示各种特征的情绪的基本可视化，并强调一些观察结果。

请注意，可视化中使用的记录来自一个公共数据集 [IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm) ，该数据集是一个说话者在不同情绪下说出同一句话。数据集是人工生成的数据集，训练有素的演员已经表演了在复杂环境中记录的情感。

## **1。各种情绪的音频时间序列:**

![](img/76754e5247bb286be0375e1204778e65.png)

图片(作者):各种情绪的原始音频时间序列可视化 [(IEMOCAP 数据集)](https://sail.usc.edu/iemocap/)

**观察:**

*   很明显，愤怒、恐惧和快乐的情绪达到了 0.5 的更高幅度，因为人们通常对这些情绪很敏感。但这也说明，这些情绪在时域上会有明显的重叠。
*   悲伤、中性、厌恶等情绪的幅度较低。梅尔谱图(n_mels = 128)
*   图片(作者):各种情绪的 Mel 谱图(n_mels = 128)可视化 [(IEMOCAP 数据集)](https://sail.usc.edu/iemocap/)

## **观察:**

![](img/9d54591a22c190c0ccaeb643273820d0.png)

**响度:** 0 dB 为参考点，最大声强。随着它变得越来越消极，声音变得越来越听不见

**音调:**频率表示振动发生的速度。这可能会受到环境因素的影响。

*   类似的观察也可以从这些可视化中得出。在恐惧和愤怒的情况下，可以观察到低噪音的呼吸声。另外，声音的强度集中在特定的时间段，音调较高。
*   在 n_mels 减少的情况下，特征和情节不反映低分贝声音。因此，对于像音频中的噪声检测这样的用例，n _ mels 应该保持在较高的一侧。
*   **3。MFCC 特征图(n_mfcc = 1)**
*   图片(作者):各种情绪的 MFCC 特征图(n_mfcc = 1)的可视化 [(IEMOCAP 数据集)](https://sail.usc.edu/iemocap/)

## **观察:**

![](img/b24321f972a52fdbf5841e877560834b.png)

对于像恐惧、快乐这样的情绪，声音水平(dB)变得非常低，我们可以观察到更多的黑色区域。

对于愤怒等情绪，中性的声音水平(dB)不会很低，一些紫色区域清晰可见。

*   这表明，像恐惧或快乐这样的情绪是冲动的，持续时间较短，而愤怒、中性和悲伤的情绪在时间线上传播得更广。
*   恐惧和愤怒情绪在 0 dB 参考点附近有更多的黄色区域，这表明与其他情绪相比，这些情绪的声音更大。
*   MFCC 特征代表了各种情绪在响度、情绪传播和频率范围方面的更好区别。这就是为什么 MFCC 特征最适合声学建模并在研究工作中广泛使用的原因。这些甚至被用作初始化深度学习架构的原始特征。
*   **4。可视化自然音频记录**

人工生成的音频记录的可视化表明，存在可用于识别情绪的某种模式。但是自然录音也是如此吗？

## 因此，我试着从自然录音中可视化真实的情绪，如下图所示。

图片(作者):各种情绪的原始音频时间序列的可视化(自然音频)

图片(作者):各种情绪(自然音频)的 Mel 声谱图(n_mels = 128)的可视化

![](img/54701c207cd236b10612857341a6c2d0.png)

**观察:**

![](img/b5b2e5293b73b0a190917b90dff07351.png)

区分时间序列的情绪和真实情绪流的频谱图变得相对困难。

因此，由于提取复杂特征的能力，与依赖于手工特征的算法相比，深度学习导致的算法进步极大地提高了 SER 系统的性能。

*   旁注:语音情感识别是一个非常具有挑战性但又非常有趣的课题。在就 SER 解决方案进行演讲和演示时，我意识到它通常非常吸引观众，原因如下-
*   人们对音频的注释有不同的看法，就像有些人可以认为音频中的愤怒的人是快乐的，反之亦然，人们最终会给出有趣的回应来证明他们的观点。

数据集中愤怒或悲伤的片段有时也很有趣。(演员为悲伤语气提及最近分手:p)

*   有时，观众会试图表现情感，从而营造一个非常吸引人的环境。
*   参考资料:
*   https://sail.usc.edu/iemocap/

[https://analyticsindiamag . com/step-by-step-to-audio-visualization-in-python/](https://analyticsindiamag.com/step-by-step-guide-to-audio-visualization-in-python/)

1.  [https://librosa . org/doc/latest/generated/librosa . feature . Mel spectrogram . html # librosa . feature . Mel spectrogram](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html#librosa.feature.melspectrogram)
2.  [https://librosa . org/doc/latest/generated/librosa . display . wave plot . html # librosa . display . wave plot](https://librosa.org/doc/latest/generated/librosa.display.waveplot.html#librosa.display.waveplot)
3.  [https://librosa . org/doc/latest/generated/librosa . display . spec show . html](https://librosa.org/doc/latest/generated/librosa.display.specshow.html)
4.  [https://librosa.org/doc/latest/generated/librosa.display.waveplot.html#librosa.display.waveplot](https://librosa.org/doc/latest/generated/librosa.display.waveplot.html#librosa.display.waveplot)
5.  [https://librosa.org/doc/latest/generated/librosa.display.specshow.html](https://librosa.org/doc/latest/generated/librosa.display.specshow.html)