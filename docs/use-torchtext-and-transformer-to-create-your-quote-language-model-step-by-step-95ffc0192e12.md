# 使用 torchtext 和 Transformer 逐步创建您的报价语言模型！

> 原文：<https://medium.com/analytics-vidhya/use-torchtext-and-transformer-to-create-your-quote-language-model-step-by-step-95ffc0192e12?source=collection_archive---------14----------------------->

![](img/eb752b6f5f2da36008f0aae7d99c70ca.png)

由 [Fab Lentz](https://unsplash.com/@fossy?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

如果你对深度学习，尤其是自然语言处理感兴趣，你可能知道文本数据的问题。我指的是你在创建模型之前就应该努力解决的问题。这些问题比如创建词汇，字符串的数值化，填充，批处理等等！有许多库可以帮助你处理这些困难的步骤，但是由于某些原因，没有一个能让我满意。一个最重要的原因是，我觉得它们并不完全兼容深度学习框架，比如 pytorch。但是现在，多亏了 pytorch，我们有了一个伟大的助手来解决我们的文本数据问题，是的 **TorchText** ！！

在这篇文章中，我想向你展示如何使用 torchtext 轻松处理你的虚拟数据。此外，我将使用 nn.transformer(最近已添加到 nn 中)来设计我的模型，以使 transformer 的优势受益。

# 我们的任务

我们有一个简单的 txt 文件，其中包含一些报价，我们想训练一个语言模型，可以生成报价！！你可以从[这里](https://github.com/mmsamiei/just-practice-deep/blob/master/lava-language-model/text.txt)下载一个示例 txt 文件！

# 我们模型的结构

我将使用一个变压器解码器来学习语言模型。记住，在这个任务中我们不需要变压器编码器。(你会看到的)

# 履行

好的，让我们一步一步来！！

```
import torchimport torch.nn as nnimport numpy as npdevice = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

如此简单！在本节中，我们刚刚导入并设置了我们的设备！

好了，现在让我们开始研究我们的模型:

```
class LM(nn.Module):

  def __init__(self, hid_size, vocab_size, n_head, n_layers, max_len, device):
    super().__init__()
    self.device = device
    self.hid_size = hid_size
    self.max_len = max_len
    self.embedding = nn.Embedding(vocab_size, hid_size)
    self.position_enc = nn.Embedding(self.max_len, self.hid_size)
    self.position_enc.weight.data =self.position_encoding_init(self.max_len, self.hid_size)
    self.scale = torch.sqrt(torch.FloatTensor([self.hid_size])).to(device)
    self.layer_norm = nn.LayerNorm(self.hid_size)
    self.decoder_layer = nn.TransformerDecoderLayer(d_model=hid_size, nhead = n_head)
    self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=n_layers, norm=self.layer_norm)
    self.fc = nn.Linear(hid_size, vocab_size)
    self._init_weights()

  def forward(self, x):
    sent_len, batch_size = x.shape[0], x.shape[1]
    memory_mask = self.generate_complete_mask(sent_len)
    tgt_mask = self.generate_triangular_mask(sent_len)
    memory = torch.zeros(1, batch_size, self.hid_size, device=self.device)
    temp = x
    temp = self.embedding(temp)
    pos = torch.arange(0,sent_len).unsqueeze(1).repeat(1,batch_size).to(self.device)
    temp_pos_emb = self.position_enc(pos)
    temp = temp * self.scale + temp_pos_emb
    temp = self.decoder(temp, memory, tgt_mask=tgt_mask)
    temp = self.fc(temp)
    return temp
def _init_weights(self):
    for p in self.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
def generate_triangular_mask(self, size):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
        return mask
def generate_sequence(self, src):
    #src = [sent_len]
    src = src.unsqueeze(1)
    #src = [sent_len, 1]
    generate_step = 0
    while generate_step < 20:
      out = self.forward(src)
      #out = [sent_len + 1, 1, vocab_size]
      out = torch.argmax(out[-1, :], dim=1) # [1]
      out = out.unsqueeze(0) #[1,1]
      src = torch.cat((src, out), dim=0)
      generate_step += 1
    src = src.squeeze(1)
    return src

def position_encoding_init(self, n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in  range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    temp = torch.from_numpy(position_enc).type(torch.FloatTensor)
    temp = temp.to(self.device)
    return temp
```

这么长但不那么难！我们一起来考察一下吧！

我们已经定义了我们模型 *LM* ，它本身就有变压器。感谢 pytorch，现在我们可以使用预构建的 transformer 模块，但不幸的是 pytorch 还没有实现位置嵌入，所以我们必须自己实现。那是关于 *__init__* 方法的，但是在 forward 中发生了什么呢？

在 forward 中，首先我们得到字符串的张量——然后我们通过“注意力是你所需要的全部”一文中介绍的公式对它应用单词和位置嵌入。然后，我们将嵌入序列提供给转换器(在这一步，我们使用 *position_encoding_init* 创建正弦位置嵌入)。

因为解码器转换器需要由编码器转换器产生的内存，而我们这里没有任何编码器，所以我们将它的内存设置为零！！加油！别担心，这不会毁了我们的模型。问为什么？因为解码器变换器本身有残差，如果你关注零向量，你得到零向量，因为你有残差连接，零与你先前的结果相加，所以你还会有先前的结果！

之后，我们需要生成三角形掩码，以使一个 vocab 的隐藏状态只关注它的左上下文。我们使用 *generate_triangular_mask* 制作这个遮罩。最后，我们实现了 *generate_sequence 方法*来完成我们的输入句子。现在让我们带着**火炬文本**继续与您见面！

```
import torchtext
from torchtext import data
import spacymy_tok = spacy.load('en')
def spacy_tok(x):
    return [tok.text for tok in my_tok.tokenizer(x)]

TEXT = data.Field(lower=True, tokenize=spacy_tok)
```

在这段代码中，我首先使用 spacy tokenizer 实现了一个 tokenizer(我在这里的工作类似于一个包装器！)，你可以看到 *spacy_tok* 是一个可以标记字符串的方法。重要的是文本，它是一个字段。Field 是 torchtext 的一个类，通过它你告诉 torchtext 如何查看你的原始数据。例如，我已经告诉 torchtext，我有一个字段文本，必须使用 spacy_tok 方法将其原始数据标记化。

```
from torchtext.datasets import LanguageModelingDatasetmy_dataset = LanguageModelingDataset("./text.txt", TEXT)
```

然后我用***LanguageModelingDataset***导入我的 txt 文件，同时在文本视图中查看它！(pytorch 中有类似于***LanguageModelingDataset 的类，以便为许多任务加载数据集，例如***tabular dataset***、***translation dataset***等等)***

```
TEXT.build_vocab(my_dataset)
```

现在，我们已经像你看到的那样简单地创造了我们的词汇！

```
def make_train_iter(batch_size, bptt_len):
  train_iter = data.BPTTIterator(
    my_dataset,
    batch_size=batch_size,
    bptt_len=bptt_len, # this is where we specify the sequence length
    device=device,
    repeat=False,
    shuffle=True)
  print(len(train_iter))
  return train_iter
```

现在我们创建一个方法，它给我们一个数据迭代器！我们通过包装 ***数据来实现。BPTTIterator*** ！太神奇了！Pytorch 有许多像 BPTTIterator 这样的迭代器，它们通过提供批量和处理过的数据来帮助你。BPTTIterator 是专门用于语言建模的。它有一个名为 bptt_len 字段，对它起关键作用。现在让我们假设你有一篇这样的文章“权力越大，责任越大！”并将其传递给 bptt，bptt_len = 3，然后它在迭代器中给出这个 src 和 target 对:

权力大->权力大责任大
->责任大！

这正是我们想要的语言模型训练。您也可以尝试其他 torchtext 迭代器。

让我们创建模型并计算其参数:

```
vocab_size = len(TEXT.vocab)
hid_size = 16
pf_size = 32
n_head = 4
n_layer= 1
model = LM(hid_size, vocab_size, n_head, n_layer, pf_size, device).to(device)def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)print(f'The model has {count_parameters(model):,} trainable parameters')
```

之后，我们实现了 NoamOpt，它是一个用于优化 transformer 中参数的优化器。前面已经介绍过 [*注意是你所需要的*](https://arxiv.org/pdf/1706.03762.pdf) )

```
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
```

之后，我们实现了 train_one_epoch 和 train 方法

```
def train_one_epoch(model,train_iter, optimizer, criterion, clip):
  epoch_loss = 0
  model.train()
  for batch in train_iter:
    optimizer.zero_grad()
    batch_text = batch.text
    batch_target = batch.target
    result = model(batch_text)
    loss = criterion(result.view(-1, result.shape[-1]), batch_target.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    epoch_loss += loss.item()
    return epoch_loss / len(train_iter)
    print("epoch is {} loss is {}".format(epoch, epoch_loss / len(train_iter)))def train(model, train_iter, optimizer, criterion, clip, N_EPOCH):
  for epoch in range(N_EPOCH):
    epoch_loss = train_one_epoch(model, train_iter, optimizer, criterion, clip)
    print("epoch is {} loss is {}".format(epoch, epoch_loss))
```

让我们训练…

```
for i in range(1, 3):optimizer = NoamOpt(hid_size, 1, 2000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))criterion = torch.nn.CrossEntropyLoss()train_iter = make_train_iter(4096, i)train(model, train_iter, optimizer, criterion, 1, 3)
```

现在，如果您想测试您的模型并生成报价，您可以使用下面的代码(编写一个表达式，以便模型完成它！)

```
source_sentence = ["your","expression","word","by","word"] ## you must write its word in an array, because i'm so tired to complete it :)))
print(source_sentence)
model.eval()
print(' '.join(source_sentence))
print()
x = TEXT.numericalize([source_sentence]).to(device).squeeze(1)
generated_sequence =model.generate_sequence(x)
words = [TEXT.vocab.itos[word_idx] for word_idx in generated_sequence]
print(' '.join(words))
```

你可以在[这里](https://github.com/mmsamiei/just-practice-deep/blob/master/lava-language-model/lava.ipynb)看到完整代码的笔记本

**这是我最后的一幕！我希望尽快给你写信！再见！**