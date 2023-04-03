# 了解新 GPT 协议源代码第 2 部分

> 原文：<https://medium.com/analytics-vidhya/understanding-the-gpt-2-source-code-part-2-4a980c36c68b?source=collection_archive---------0----------------------->

![](img/e65a421434f0bd70afc8c5a9e1754d1c.png)

嗨！这是试图理解 GPT-2 的源代码的系列文章的下一篇，希望能学到一些东西。第 1 部分可以在这里找到[。如果有任何问题，不清楚的地方或反馈，请不要犹豫，在评论中提出来！](/@isamu.website/understanding-the-gpt-2-source-code-part-1-4481328ee10b)

在这一部分，我将浏览 encoder.py 和 encode.py。

# 什么是编码？

需要理解的最重要的事情之一是，当您向模型中输入文本时，它不能仅仅使用该文本。在训练之前，机器对什么是“苹果”或“梨”以及它们之间的关系毫无概念。事实上，对于机器来说，出现“苹果”或“梨”这两个词会让它彻底糊涂。它宁愿看到像 1 和 2 这样的数字来代表它们。这就是编码的作用！它能把单词转换成数字！

# OpenAI 是怎么做到的？

首先我们来看看 encode.py，内容给出如下。

```
#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/output.npz
#  PYTHONPATH=src ./train --dataset /path/to/output.npzimport argparse
import numpy as npimport encoder
from load_dataset import load_datasetparser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('in_text', metavar='PATH', type=str, help='Input file, directory, or glob pattern (utf-8 text).')
parser.add_argument('out_npz', metavar='OUT.npz', type=str, help='Output file path')def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    print('Reading files')
    chunks = load_dataset(enc, args.in_text, args.combine)
    print('Writing', args.out_npz)
    np.savez_compressed(args.out_npz, *chunks)if __name__ == '__main__':
    main()
```

如解析器所示，encode.py 接受 4 个参数。我不知道他们为什么不使用这里的消防图书馆，所以如果有人知道，请告诉我！

这四个参数是

*   model_name —据我所知，目前只有 117M 和 345M。
*   组合——写着“用分隔符将文件连接成这个最小大小的块”。我现在不明白。因此，我计划进一步深入源代码，以便找出这个参数具体做什么。
*   in _ text 输入。txt 文件
*   out _ npz-npz 格式的输出文件。

我们接下来看到的第一条有趣的线是

```
enc = encoder.get_encoder(args.model_name)
```

因此，让我们研究一下 encoder.py，看看 get_encoder 函数做了什么。

```
def get_encoder(model_name):
    with open(os.path.join('models', model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join('models', model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
```

该函数做 3 件事。

1.  获取 encoder.json
2.  获取 vocab.bpe 并在新行处拆分，忽略第一个和最后一个字符
3.  Initialize 返回用 encoder.json 和 vocab.bpe 初始化的编码器类

117M 型号和 345M 型号的 encoder.json 和 vocab.bpe 相同，因此型号名称没有那么重要。当您打开 encoder.json 并查看其内容时，您会看到

```
{“!”: 0, “\””: 1, “#”: 2, “$”: 3, “%”: 4, “&”: 5, “‘“: 6, “(“: 7, “)”: 8, “*”: 9, “+”: 10, “,”: 11, “-”: 12,
```

依此类推，直到

```
“\u0120Collider”: 50253, “\u0120informants”: 50254, “\u0120gazed”: 50255, “<|endoftext|>”: 50256}
```

因此，很明显，这个 encoder.json 表示每个单词或符号映射到的数字。

但是，对于 vocab.bpe，打开后看到的是

```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġt he 
```

我不知道这是关于什么的。

# vocab.bpe 是做什么的？

显然，这是一种叫做字节对编码的东西。根据维基百科的说法，这是一种压缩技术，用那里的例子来说，给定一个字符串

```
aaabdaaabac
```

由于 aa 重复不止一次，我们可以用一个未使用的字节来替换它，Z 它可以压缩为

```
ZabdZabac
Z=aa
```

由于 ab 重复，所以可以用 Y 替换为

```
ZYdZYac
Y=ab
Z=aa
```

依此类推，直到没有重复的字节对。然而，从文件来看，似乎至少有一个轻微的修改作为字符，例如“h”，我怀疑是未使用的，用于表示单个字母“e”，而不是算法似乎建议的一对字符。

由于这方面的源代码不太可用，我决定在网上搜索！我首先发现的是这张[纸](https://www.aclweb.org/anthology/P16-1162)。TLDR，这基本上是关于如何字节对编码可以用来找到新词的意义。

举一个例子，在论文中给出了新的单词

德语中的“Abwasser | behandlungs | anlange ”,如果我们使用字节对编码，它可以被分割成 3 个子词“污水处理厂”,而如果我们只是从一开始就将其编码成一个向量，在遇到它时，没有办法说出它是关于什么的。

然而，我仍然对这样的序列感到困惑

```
Ġ t
Ġ a
h e
i n
r e
o n
Ġt h
```

似是而非。因此，我决定在网上实施，这里可以找到。谢谢，瑞科·森里奇！我查看了代码，我的理解是，并不是ġ对应于 t，而是ġt 是字节对！

冒着有点无聊的风险，我想我会解释一下。如果不感兴趣，请到下一节！

所以，我基本上做的是搜索输出文件中提到的任何内容。我发现在 learn_bpe 函数中提到了它。有两个例子。第一个是

```
outfile.write(‘#version: 0.2\n’)
```

第二个是

```
outfile.write(‘{0} {1}\n’.format(*most_frequent))
```

这个版本非常明显地表明 OpenAI 使用了这个 python 文件！最频繁基本上是轮流获取最频繁的字节对，并将它们追加写入 outfile。

# 编码数据集

encode.py 中的下一行(不是注释)是

```
chunks = load_dataset(enc, args.in_text, args.combine)
```

这里的 enc 是先前返回的编码器实例。现在，让我们看看 load_dataset 函数。第一部分是

```
def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)
```

这基本上是将目录中一个或多个文本文件的路径附加到一个名为 paths 的列表中。os.walk 是一个奇特的函数，可以遍历目录中的文件。Globs 基本上是带有通配符的文件。比如，*。txt 可以是任何形式的文本文件，如 a.txt、adfdj.txt 等，因为*是一个特殊的通配符。因此，它是一个球体。如果我错了，请在评论中告诉我！

```
token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks
```

第一部分是。

```
if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
```

用于已经编码的文件。基本上，它所做的就是用这个新编码的文件覆盖输出文件。

```
else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
```

在这里，我终于明白了组合参数的含义。如果文本少于 combine 的字符数，文本文件将被忽略。现在，让我们通过查看编码器编码的方法来看看 enc.encode(raw_text)是做什么的。

```
def encode(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens
```

TLDR；

这里发生的最基本的事情是

1.  对于文本中的每个模式，将该模式作为令牌返回
2.  将令牌编码成 utf-8 格式，并连接成一个名为 token 的字符串
3.  扩展 bpe 令牌数组，在令牌中包含字节对，即字符对。

对于那些好奇的人来说，

为了理解这一点，让我们从头来过函数。下面是第一行有趣的内容。

```
for token in re.findall(self.pat, text):
```

self.pat 在哪里，

```
self.pat = re.compile(r”””’s|’t|’re|’ve|’m|’ll|’d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+”””)
```

# 对此正则表达式的解释

re.compile 只是用来预编译字符串的。现在让我们来看看这根弦。这个字符串叫做[正则表达式](https://docs.python.org/3/library/re.html)。基本上，它所做的是表示模式文本。例如，如果模式由单个单词“a”组成，并且如果我们试图在文本“我吃了很多馅饼”中找到所有的模式，那么该模式将只出现在两个实例中:在“had”和“a”中。

关于这个字符串，首先要注意的是其中有很多“|”。如果你有使用大多数编程语言的经验，除了 python，我敢肯定你会知道它的意思是或者。

问号对应于前一个字符的 0 或 1 次重复。那么，对于“？”它基本上接受任意数量的空格。

按[这里的](https://stackoverflow.com/questions/14891129/regular-expression-pl-and-pn)，

> `\p{L}`匹配类别“字母”中的单个代码点。
> `\p{N}`匹配任何脚本中的任何种类的数字字符。

但是，这里需要注意的是，这不在 python 的 re 库中，只在 regex 库中可用。所以，OpenAI 团队从写作开始

```
import regex as re
```

因为“+”表示一个或多个，\p{L}+可以匹配任何单词，\p{N}+可以匹配任何数字

\s 匹配任何 Unicode 空白字符，包括\n 等。While \S 匹配任何非空白字符。^是新的路线和？！意味着如果它前面的图案不出现，它将放大它前面的图案。

所以，基本上，它所做的就是把像“他们”这样的词分成“他们”和“他们”等等。

# 字节编码器的解释

```
token = ‘’.join(self.byte_encoder[b] for b in token.encode(‘utf-8’))
```

这里，令牌模式被编码成 utf-8 格式。然后在 for b in 循环中，改成函数 ord 给的数。

例如，order(" a "。encode('utf-8 '))给出 97 而

```
[b for b in “a”.encode(‘utf-8’)] 
```

也给出了 97。据我所知，byte_encoder 在某些情况下会返回稍加修改的编码 unicode。通过稍微修改，我的意思是 2⁸.的数字跳跃 self.byte_encoder 用 bytes_to_unicode 函数初始化。以下是哪一个

```
[@lru_cache](http://twitter.com/lru_cache)()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
```

如果我理解正确的话，lru_cache 是一个来自 functools 模块的装饰函数，它缓存函数的结果，这样即使你第二次调用一个函数，也不需要进行不必要的处理。但是因为这个函数只被调用一次，所以我认为没有理由使用这个装饰器。如果有人能解释请告诉我！

因为我认为代码是不言自明的，我将跳过它，除了提到 chr()与 ord()相反，当你给它一个数字时，它返回一个字符。因此，该函数返回一个字典，其中包含从 1 到 2⁸的键以及相应的字符。

我无法理解这些评论，但我认为 2⁸转变的部分原因如下。

```
And avoids mapping to whitespace/control characters the bpe code barfs on.
```

但坦率地说，我并不完全理解它。然而，总体效果是令牌被转换成合适的格式。

# 对字节对编码标记化的解释

```
bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
```

首先，让我们看看 self.bpe 函数。一开始是这样的

```
if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
```

get_pairs 基本上将每个字符对配对并返回。

```
while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
```

在这里，首先要注意的是 self.bpe_ranks。这是由定义的

```
self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
```

在 init 函数中。其中 bpe_merges 是 vocab.bpe 给出的数组，因此，最频繁的字节对被赋予最小的数字，最不频繁的被赋予最大的数字。因此，当我们看下面这条线时，

```
bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
```

显而易见，二元模型表示数据集中整个词汇表中最常见的字符对。float('inf ')意味着如果在 vocab.bpe 的 bpe_ranks 中没有找到该对，则返回 infinity。由于无穷大不可能是最小值，所以干脆丢弃。

```
try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
```

为了。index 函数，第二个参数表示范围的起始索引，而第一个参数表示在元组字中搜索的值。如果我没有超过 word 中给定的所有值，那么首先在 new_word 中添加 word[i:j]。

如果我确实超出了限制，那么将引发一个错误，except 将捕获它，从 I 到结尾的所有字符都将添加到 new_word 中。

现在，更重要的是，

```
if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
```

在这里，成对的单词也被添加到 new_word 中！最后，

```
new_word = tuple(new_word)
word = new_word
if len(word) == 1:
   break
else:
   pairs = get_pairs(word)
```

在这里，我们看到循环将继续下去，直到单词的长度变为 1，以及 word 被赋予 new_word。现在，在下一个循环中，如果我们回想一下字节对编码是怎么回事，现在就有可能找到 2 个字母字符和另一个 2 个字母字符或 1 个字母字符之间的字节对或字符对。在下一个循环中，甚至可以产生更长的字节对，因为它总是保证选择最常见的字节对，因为下面的行！

```
bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
```

现在，字符的长度没有变成一也是很有可能的。然而，在 while true 循环中还有一个 break 语句，那就是

```
if bigram not in self.bpe_ranks:
 break
```

因此，可以有把握地说，如果 word 中没有更多的有效字节对，而 word 又不能被简化为更小的标记，那么循环将会终止。

然后，

```
word = ‘ ‘.join(word)
 self.cache[token] = word
 return word
```

单词由一个空格连接，然后返回。

回到编码功能，

```
bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(‘ ‘))
```

bpe 令牌由 self.encoder 转换成数字，self . encoder 是一个以前加载的 json 文件。

抱歉，如果以上解释变得有点复杂。我不确定我是否完全理解了这里的所有代码。特别是 try/except 块，所以如果有人可以指出我的错误或不清楚，请说出来。

# 保存到输出

回到 load_dataset.py，

```
if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
return token_chunks
```

输出的标记被转换成 numpy 数组并附加到标记上，最后在 encode.py 中

```
np.savez_compressed(args.out_npz, *chunks)
```

输出按原样保存。

# 关于数据我们需要小心的事情

正如我们从编码过程中看到的，end_tokens 不会自动添加到训练数据中。因此，最好建议在使用您自己的定制数据集来微调数据时，在文本的末尾提供结束标记，尤其是在文本很短的情况下！

# 然后

在下一个故事中，我将尝试进入 sample.py 和 model.py！如果你有兴趣，请在这里阅读[！](/@isamu.website/understanding-the-gpt-2-source-code-part-3-9796a5a5cc7c)