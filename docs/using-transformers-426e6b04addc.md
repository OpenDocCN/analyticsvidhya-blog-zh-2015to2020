# 如何开始使用 NLP 的变压器

> 原文：<https://medium.com/analytics-vidhya/using-transformers-426e6b04addc?source=collection_archive---------28----------------------->

当你可以自己尝试不同的方法时，这篇文章效果最好——在[deepnote.com](http://deepnote.com/)上运行我的笔记本来试试吧！

我喜欢变形金刚图书馆。这是迄今为止最容易开始使用变压器的自然语言处理，这是目前的出血边缘。

第一步是从 transformers 库中获取模型和标记器

```
import tensorflow as tf 
import tensorflow_hub as hub
from transformers.modeling_tf_openai import TFOpenAIGPTLMHeadModel #this is the GPT transformer with additional layers added for easy language modelingfrom transformers.tokenization_openai import OpenAIGPTTokenizerimport simpletransformers
```

如果下面的单元格需要一些时间，也不用担心！模型下载只需要一分钟。

```
model = TFOpenAIGPTLMHeadModel.from_pretrained("openai-gpt")tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
```

注意:GPT2 是用于文本生成的较新的 transformer 模型，但是当我编写这段代码时，只有 GPT 支持文本生成

```
prompt_text = "<Think of a piece of text to supply as a prompt!>"encoded_prompt = tokenizer.encode(prompt_text,add_special_tokens=False,return_tensors='tf')
```

由于 transformers 为 GPT 模型提供了一个标记器，这是一个比使用通用句子编码器更容易的解决方案。tensors 返回 pytorch 类型，而不是 tensorflow，因为我发现在这种情况下遇到的 bug 更少(这可能是库最初是为 pytorch 编写的结果)。

```
num_sequences = 1 # this is the number of different sequences/ sentences the generator will create given the promptlength = 15 # if you want to strictly follow the prompt for this question set the length to one. If you want to see how creative GPT can be feel free to amend
```

来点序列吧！

```
generated_sequences = model.generate(
input_ids=encoded_prompt,
do_sample=True,
max_length=length + len(encoded_prompt[0]),
temperature=1.0,
top_k=5,
top_p=0.9,
repetition_penalty=1.0)
```

牢房里有很多东西要打开。该表扬的地方要表扬——图书馆有很棒的文档，所以让我们打开包装吧。

输入 id 只是简单地指定输入是什么。

Do_sample —这可以防止模型在每一步都贪婪地挑选最可能的单词。

Max_len —我们想要的序列长度是多少？

温度——我发现这个非常有趣。这是衡量我们的模型在选词时会有多大风险的一个尺度。请随意调整这个！

Top_k 和 top_p 是相似的，因为它们限制了模型在从单词概率中随机采样之前解码时考虑的单词数量。

重复惩罚是为了避免句子重复，没有任何真正有趣的东西。

感谢[医生](https://huggingface.co/transformers/main_classes/model.html?highlight=tfpretrained#transformers.TFPreTrainedModel)和[这篇](/voice-tech-podcast/visualising-beam-search-and-other-decoding-algorithms-for-natural-language-generation-fbba7cba2c5b)优秀的媒体文章帮助我理解了这里的事实。

对于 generated_sequences 中的序列:

```
for sequence in generated_sequences:text = tokenizer.decode(sequence, clean_up_tokenization_spaces=True)print(text)
```

还有 tada！我们已经预测了文本。然而，这并不是生成文本的唯一方式。到目前为止，最简单的方法是使用管道。

```
from transformers import pipelinegenerator = pipeline("text-generation")text_1 = generator("Text generation is cool because", max_length=50)text_1[0]['generated_text']
```

对于本文的其余部分，请点击[这个](https://deepnote.com/project/e34c35b3-eeb8-4706-962c-57341bd0dafe#%2Ftext_generation.ipynb)链接查看 deepnote 的完整版本