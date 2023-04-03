# 将张量流模型转换为变压器模型

> 原文：<https://medium.com/analytics-vidhya/this-article-shows-how-to-convert-the-tensorflow-model-to-the-huggingface-transformers-model-4a39a09a2242?source=collection_archive---------9----------------------->

本文展示了如何将 **Tensorflow 模型**转换为****拥抱脸变形金刚模型**。我将使用 Bert Tensorflow 模型和 BioBERT TensorFlow 模型作为示例，并将它们转换为基于 PyTorch 的 HuggingFace transformer，以便无缝使用。我还将展示一个代码示例，它将展示转换后的模型可以像任何其他 Transformer 模型一样被处理。
完整的代码可以在[本笔记本](https://colab.research.google.com/drive/1ezVDyBXsD1GutXQfQc0tqRrSvZlHxN0u?usp=sharing)中找到**

****什么是 BioBERT:**[**https://github.com/dmis-lab/biobert**](https://github.com/dmis-lab/biobert)**？这是一个基于…的语言表示模型****