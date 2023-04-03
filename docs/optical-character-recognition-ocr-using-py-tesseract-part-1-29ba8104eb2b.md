# 使用(Py)宇宙魔方的光学字符识别(OCR ):第 1 部分

> 原文：<https://medium.com/analytics-vidhya/optical-character-recognition-ocr-using-py-tesseract-part-1-29ba8104eb2b?source=collection_archive---------22----------------------->

Python-tesseract 是 Python 的光学字符识别(OCR)工具。也就是说，它将识别并“读取”嵌入图像中的文本。

Python-tesseract 是 Google 的 Tesseract-OCR 引擎的包装器。它作为 tesseract 的独立调用脚本也很有用，因为它可以读取 Pillow 和 Leptonica 图像库支持的所有图像类型，包括 jpeg、png、gif、BMP、tiff 等。此外，如果用作脚本，Python-tesseract 将打印识别的文本，而不是将其写入文件。