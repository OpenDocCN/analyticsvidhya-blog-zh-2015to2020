# 艾写诗:用火炬构筑模式

> 原文：<https://medium.com/analytics-vidhya/ai-writing-poems-building-lstm-model-using-pytorch-d1c58a24bb64?source=collection_archive---------8----------------------->

![](img/948265820c2fc9979798748bebc50d6a.png)

大家好！！在本文中，我们将使用 PyTorch 构建一个模型来预测段落中的下一个单词。首先，我们将了解 RNN 和 LSTM，以及他们是如何工作的。然后我们将创建我们的模型。首先，我们加载数据并对其进行预处理。然后我们将使用 PyTorch 来训练模型并保存它。在那之后，我们将通过给它一个起始文本来从该模型中进行预测，并使用它来生成完整的段落。

# 什么是 RNN？

在机器学习中，像对狗或猫的图像进行分类这样的简单问题可以通过简单地对数据集进行训练并使用分类器来解决。但是如果我们的问题更复杂，比如我们必须预测段落中的下一个单词。因此，如果我们仔细研究这个问题，我们会发现，我们人类不能仅仅通过使用我们现有的语言和语法知识来解决这个问题。在这种类型的问题中，我们使用段落的前一个单词和段落的上下文来预测下一个单词。

传统的神经网络无法做到这一点，因为它们是在固定的数据集上训练的，然后用来进行预测。RNN 被用来解决这类问题。RNN 代表**递归神经网络**。我们可以把 RNN 想象成一个有回路的神经网络。它将一个状态的信息传递给下一个状态。因此，信息在这个过程中持续存在，我们可以利用它来理解之前的背景，并做出准确的预测。

# 什么是 LSTM？

因此，如果我们通过使用 RNN 来解决数据序列的问题，而以前的上下文是被使用的，那么我们为什么需要 LSTM 呢？要回答这个问题，我们必须看看这两个例子。

**例一**:“鸟住在*巢*。”这里很容易预测单词“nest ”,因为我们已经有了 bird 的上下文，RNN 在这种情况下会工作得很好。

**例 2:**

“我在印度长大……..所以我会说 H *indi* ”所以这里预测单词“Hindi”的任务对于一个 RNN 人来说是困难的，因为这里语境之间的差距很大。通过看“我能说……”这一行，我们无法预测语言，我们将需要额外的印度背景。所以在这里，我们需要对我们的段落有一些长期的依赖，这样我们才能理解上下文。

为此，我们使用 LSTM( **长短期记忆**)。顾名思义，它们有长时记忆和短时记忆(gate ),两者都用在连词中来做预测。如果我们谈论 LSTM 的建筑，它们包含 4 个门，即学习门、忘记门、记忆门、使用门。为了让这篇文章简单易懂，我不打算深入 LSTM 的建筑理论。但是也许我们会在接下来的文章中讨论它。(也许在下一部😉).

# 让我们建立我们的模型。

现在我们已经完成了理论，让我们开始有趣的部分——建立我们的模型。

# 加载和预处理数据

我将使用来自 Kaggle 的诗歌数据集。它总共有 15，000 首诗，因此对于我们模型学习和创建模式来说足够了。现在让我们开始把它载入我们的笔记本。

1.首先导入库。

```
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
```

2.现在从文本文件加载数据。

```
# open text file and read in data as `text`
with open('/data/poems_data.txt', 'r') as f:
    text = f.read()
```

3.我们可以通过打印前 100 个字符来验证我们的数据。

```
text[:100]
```

4.正如我们所知，我们的神经网络不理解文本，所以我们必须将我们的 txt 数据转换为整数。为此，我们可以创建令牌字典，并将字符映射到整数，反之亦然。

```
# encode the text and map each character to an integer and vice versa# we create two dictionaries:
# 1\. int2char, which maps integers to characters
# 2\. char2int, which maps characters to unique integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}# encode the text
encoded = np.array([char2int[ch] for ch in text])
```

5.一种热编码用于表示字符。比如，如果我们有三个字符 a，b，c，那么我们可以这样来表示它们[1，0，0]，[0，1，0]，[0，0，1]这里我们用 1 来表示这个字符，其他的都是 0。对于我们的用例，我们有许多字符和符号，所以我们的一个热点向量会很长。但是没关系。

```
def one_hot_encode(arr, n_labels):

    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot
```

现在用这种方法测试它。

```
# check that the function works as expected
test_seq = np.array([[0, 5, 1]])
one_hot = one_hot_encode(test_seq, 8)print(one_hot)
```

6.现在我们必须为我们的模型创建批次，这是非常关键的部分。在这种情况下，我们将选择一个批量大小，即行数，然后序列长度是一个批量中要使用的列数。

```
def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    batch_size_total = batch_size * seq_length
    # total number of batches we can make
    n_batches = len(arr)//batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
```

7 .现在我们可以检查 GPU 是否可用。(如果 GPU 不可用，请保持较低的 epochs 数)

```
# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')
```

8.这里我们创建了一个名为 CharRNN 的类。这是我们的模特班。在 init 方法中，我们必须为我们的模型定义层。这里我们使用两个 LSTM 层。我们还使用了 dropout(这有助于避免过度拟合)。对于输出，我们使用简单的线性层。

```
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        #lstm layer
        self.lstm=nn.LSTM(len(self.chars),n_hidden,n_layers,
                          dropout=drop_prob,batch_first=True)

        #dropout layer
        self.dropout=nn.Dropout(drop_prob)

        #output layer
        self.fc=nn.Linear(n_hidden,len(self.chars)) def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        ## Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## put x through the fully-connected layer
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
```

9.现在我们有了模型，是时候训练模型了。对于训练，我们必须使用优化器和损失函数。我们简单地计算每一步后的损失，然后优化器阶跃函数将其反向传播，并适当地改变权重。损失会慢慢减少，这意味着我们的模式正在变得更好。

我们还使用两个时期之间的验证来获得验证损失，因为这样我们就可以决定我们的模型是欠拟合还是过拟合。

```
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network 

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if(train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda() # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h]) # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda() output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())

                    val_losses.append(val_loss.item())

                net.train() # reset to train mode after iterationg through validation data

                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))
```

现在用下面的方法训练它。

```
# define and print the net
n_hidden = 512
n_layers = 2net = CharRNN(chars, n_hidden, n_layers)
print(net)batch_size = 128
seq_length = 100
n_epochs =  10# start small if you are just testing initial behavior# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)
```

10.我们可以用下面的方法保存模型。

```
# change the name, for saving multiple files
model_name = 'poem_4_epoch.net'checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)
```

11.既然模型已经训练好了，我们就要从中取样并预测下一个角色！为了采样，我们传入一个字符，让网络预测下一个字符。然后我们把那个字符传回来，得到另一个预测的字符。只要继续这样做，你就会生成一堆文本！这里，前 k 个样本只是我们的模型将预测并从中使用最相关的一个字母的数量。

```
def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''

        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)

        if(train_on_gpu):
            inputs = inputs.cuda()

        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = net(inputs, h) # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu

        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())

        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], hdef sample(net, size, prime='The', top_k=None):

    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval() # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k) chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char) return ''.join(chars)
```

12.现在我们可以用这个样本方法来做预测。

```
print(sample(net, 500, prime='christmas', top_k=2))
```

输出看起来会像这样。

```
christmas a son of thisthe sun wants the street of the stars, and the way the way
they went and too man and the star of the words
of a body of a street and the strange shoulder of the skyand the sun, an end on the sun and the sun and so to the stars are stars
and the words of the water and the streets of the world
to see them to start a posture of the streets
on the street of the streets, and the sun and soul of the station
and so too too the world of a sound and stranger and to the world
to the sun a
```

正如我们所看到的，我们的模型能够生成一些好的线条。内容没有太多意义，但它能够生成一些语法正确的行。如果我们能多训练它一段时间，它会表现得更好。

# 结论

在这篇文章中，我们了解了 RNN 和 LSTM。我们还用 PyTorch 建立了我们的诗歌模型。我希望这篇文章对你有所帮助。如果您有任何疑问或建议，欢迎在下面的评论区发表，或者通过[yash.yn59@gmail.com](mailto:yash.yn59@gmail.com)联系我，我将非常乐意为您提供帮助。