# 理解伯特建筑

> 原文：<https://medium.com/analytics-vidhya/understanding-bert-architecture-3f35a264b187?source=collection_archive---------1----------------------->

![](img/eca4196e1fdf8ae9501a722d1bea18bb.png)

安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

[BERT](https://arxiv.org/pdf/1810.04805.pdf) 可能是近年来 NLP 中最令人兴奋的发展之一。就在上个月，甚至谷歌也宣布在其搜索中使用 BERT，这应该是过去五年中它在理解搜索方面的“最大飞跃”。这是来自谷歌的一个巨大证明。关于搜索！这就是伯特的重要性。

现在有一些惊人的资源可以详细了解 BERT、变形金刚和注意力网络(注意力和变形金刚是 BERT 的积木)。我在脚注中把它们联系起来。**这篇文章是关于更好地理解架构和参数**，*一旦你已经相当好地理解了 BERT*。

幸运的是，使用 Keras(和 keras_bert ),这个模型很容易加载到 Python 中。下面的代码加载模型，并打印所有层的摘要。

```
import keras
from keras_bert import get_base_dict, get_model, compile_model, gen_batch_inputs# Build & train the model
model = get_model(
    token_num=30000,
    head_num=12,
    transformer_num=12,
    embed_dim=768,
    feed_forward_dim=3072,
    seq_len=512,
    pos_num=512,
    dropout_rate=0.05,
)
compile_model(model)
model.summary()
```

我用过的所有参数，包括令牌数，这里都是来自 BERT base 模型(BERT 有两个变体，一个小变体叫 base，另一个叫 large)。这些参数是(记住符号，因为我们稍后将使用它们):

1.  令牌数(T) = 30k。这是从单词块标记化派生的不同标记的编号。这将单个单词分解成组成单词，以提高覆盖率。例:播放转换为(播放，##ing)。因此，只要模型知道“睡眠”这个词，它就可以推断出“睡眠”的含义，即使它是第一次看到这个词
2.  head_num (A) = 12。每个变压器层总共 12 个注意头
3.  变压器数量(L) = 12
4.  embed_dim (H) =嵌入长度=768
5.  前馈 Dim (FFD) = H*4 =3072
6.  seq_len (S)=输入句子中可以包含的最大标记数= 512
7.  pos_num (P) =要编码的位置= S = 512

这应该会给你一个很长的关于 BERT 中所有层的总结，看起来像这样:

```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input-Token (InputLayer)        (None, 512)          0                                            
__________________________________________________________________________________________________
Input-Segment (InputLayer)      (None, 512)          0                                            
__________________________________________________________________________________________________
Embedding-Token (TokenEmbedding [(None, 512, 768), ( 23040000    Input-Token[0][0]                
__________________________________________________________________________________________________
Embedding-Segment (Embedding)   (None, 512, 768)     1536        Input-Segment[0][0]              
__________________________________________________________________________________________________
Embedding-Token-Segment (Add)   (None, 512, 768)     0           Embedding-Token[0][0]            
                                                                 Embedding-Segment[0][0]          
__________________________________________________________________________________________________
Embedding-Position (PositionEmb (None, 512, 768)     393216      Embedding-Token-Segment[0][0]    
__________________________________________________________________________________________________
Embedding-Dropout (Dropout)     (None, 512, 768)     0           Embedding-Position[0][0]         
__________________________________________________________________________________________________
Embedding-Norm (LayerNormalizat (None, 512, 768)     1536        Embedding-Dropout[0][0]          
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 512, 768)     2362368     Embedding-Norm[0][0]             
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-1-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 512, 768)     0           Embedding-Norm[0][0]             
                                                                 Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-1-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-1-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-1-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-1-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-1-MultiHeadSelfAttention-
                                                                 Encoder-1-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-1-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-1-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-1-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-2-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-1-FeedForward-Norm[0][0] 
                                                                 Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-2-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-2-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-2-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-2-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-2-MultiHeadSelfAttention-
                                                                 Encoder-2-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-2-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-2-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-2-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-3-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-2-FeedForward-Norm[0][0] 
                                                                 Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-3-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-3-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-3-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-3-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-3-MultiHeadSelfAttention-
                                                                 Encoder-3-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-3-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-3-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-3-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-4-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-3-FeedForward-Norm[0][0] 
                                                                 Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-4-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-4-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-4-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-4-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-4-MultiHeadSelfAttention-
                                                                 Encoder-4-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-4-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-4-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-4-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-5-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-4-FeedForward-Norm[0][0] 
                                                                 Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-5-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-5-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-5-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-5-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-5-MultiHeadSelfAttention-
                                                                 Encoder-5-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-5-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-5-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-5-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-6-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-5-FeedForward-Norm[0][0] 
                                                                 Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-6-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-6-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-6-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-6-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-6-MultiHeadSelfAttention-
                                                                 Encoder-6-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-6-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-6-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-6-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-7-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-6-FeedForward-Norm[0][0] 
                                                                 Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-7-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-7-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-7-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-7-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-7-MultiHeadSelfAttention-
                                                                 Encoder-7-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-7-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-7-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-7-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-8-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-7-FeedForward-Norm[0][0] 
                                                                 Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-8-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-8-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-8-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-8-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-8-MultiHeadSelfAttention-
                                                                 Encoder-8-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-8-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-8-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 512, 768)     2362368     Encoder-8-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-9-MultiHeadSelfAttention[
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 512, 768)     0           Encoder-8-FeedForward-Norm[0][0] 
                                                                 Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-MultiHeadSelfAttentio (None, 512, 768)     1536        Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-FeedForward (FeedForw (None, 512, 768)     4722432     Encoder-9-MultiHeadSelfAttention-
__________________________________________________________________________________________________
Encoder-9-FeedForward-Dropout ( (None, 512, 768)     0           Encoder-9-FeedForward[0][0]      
__________________________________________________________________________________________________
Encoder-9-FeedForward-Add (Add) (None, 512, 768)     0           Encoder-9-MultiHeadSelfAttention-
                                                                 Encoder-9-FeedForward-Dropout[0][
__________________________________________________________________________________________________
Encoder-9-FeedForward-Norm (Lay (None, 512, 768)     1536        Encoder-9-FeedForward-Add[0][0]  
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 512, 768)     2362368     Encoder-9-FeedForward-Norm[0][0] 
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-9-FeedForward-Norm[0][0] 
                                                                 Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-MultiHeadSelfAttenti (None, 512, 768)     1536        Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-FeedForward (FeedFor (None, 512, 768)     4722432     Encoder-10-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-10-FeedForward-Dropout  (None, 512, 768)     0           Encoder-10-FeedForward[0][0]     
__________________________________________________________________________________________________
Encoder-10-FeedForward-Add (Add (None, 512, 768)     0           Encoder-10-MultiHeadSelfAttention
                                                                 Encoder-10-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-10-FeedForward-Norm (La (None, 512, 768)     1536        Encoder-10-FeedForward-Add[0][0] 
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 512, 768)     2362368     Encoder-10-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-10-FeedForward-Norm[0][0]
                                                                 Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-MultiHeadSelfAttenti (None, 512, 768)     1536        Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-FeedForward (FeedFor (None, 512, 768)     4722432     Encoder-11-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-11-FeedForward-Dropout  (None, 512, 768)     0           Encoder-11-FeedForward[0][0]     
__________________________________________________________________________________________________
Encoder-11-FeedForward-Add (Add (None, 512, 768)     0           Encoder-11-MultiHeadSelfAttention
                                                                 Encoder-11-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-11-FeedForward-Norm (La (None, 512, 768)     1536        Encoder-11-FeedForward-Add[0][0] 
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 512, 768)     2362368     Encoder-11-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 512, 768)     0           Encoder-11-FeedForward-Norm[0][0]
                                                                 Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-MultiHeadSelfAttenti (None, 512, 768)     1536        Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-FeedForward (FeedFor (None, 512, 768)     4722432     Encoder-12-MultiHeadSelfAttention
__________________________________________________________________________________________________
Encoder-12-FeedForward-Dropout  (None, 512, 768)     0           Encoder-12-FeedForward[0][0]     
__________________________________________________________________________________________________
Encoder-12-FeedForward-Add (Add (None, 512, 768)     0           Encoder-12-MultiHeadSelfAttention
                                                                 Encoder-12-FeedForward-Dropout[0]
__________________________________________________________________________________________________
Encoder-12-FeedForward-Norm (La (None, 512, 768)     1536        Encoder-12-FeedForward-Add[0][0] 
__________________________________________________________________________________________________
MLM-Dense (Dense)               (None, 512, 768)     590592      Encoder-12-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
MLM-Norm (LayerNormalization)   (None, 512, 768)     1536        MLM-Dense[0][0]                  
__________________________________________________________________________________________________
Extract (Extract)               (None, 768)          0           Encoder-12-FeedForward-Norm[0][0]
__________________________________________________________________________________________________
MLM-Sim (EmbeddingSimilarity)   (None, 512, 30000)   30000       MLM-Norm[0][0]                   
                                                                 Embedding-Token[0][1]            
__________________________________________________________________________________________________
Input-Masked (InputLayer)       (None, 512)          0                                            
__________________________________________________________________________________________________
NSP-Dense (Dense)               (None, 768)          590592      Extract[0][0]                    
__________________________________________________________________________________________________
MLM (Masked)                    (None, 512, 30000)   0           MLM-Sim[0][0]                    
                                                                 Input-Masked[0][0]               
__________________________________________________________________________________________________
NSP (Dense)                     (None, 2)            1538        NSP-Dense[0][0]                  
==================================================================================================
Total params: 109,705,010
Trainable params: 109,705,010
Non-trainable params: 0
```

这是一份令人生畏的清单。正如 BERT 论文中提到的，可训练参数的总数约为 110 万。这让人放心，我们加载的模型是正确的。

同样的情况也可以在图像中可视化，这有助于我们更好地理解计算图:

```
from keras.utils import plot_model
plot_model(model, to_file='bert.png')
```

![](img/5e16d890b397d1c9580594058142a49c.png)

下面是模型中各个步骤的简要介绍:

1.  两个输入:一个来自单词标记，一个来自段层
2.  这些相加，加起来就是第三次嵌入:位置嵌入，接着是丢失和层标准化
3.  然后开始多头自我关注层-每组有 9 个步骤(在上面的图像中，所有单元都以编码器-1 开始)，共有 12 个这样的层。这里面的 108 行就是为了捕捉这些。如果我们更好地理解这些，我们几乎完全理解了架构
4.  在这 12 层之后，有两个输出——一个用于 NSP(下一句预测),一个用于 MLM(屏蔽语言建模)

**分层核算:**

从上到下浏览各层，我们可以看到以下内容:

1.  输入-令牌和段不像预期的那样具有任何可训练参数。
2.  令牌嵌入参数= 23040000 (H * T) —因为 30k (T)个令牌中的每一个都需要维度 768 (H)中的表示
3.  段嵌入参数= 1536 (2*H ),因为我们需要两个长度为(H)的向量。向量分别代表线段 A 和线段 B
4.  位置嵌入增加了标记嵌入和段嵌入。参数= 393216 (H*P)。这是因为它需要为从 1 到 512 (P)开始的记号生成 P 个向量，每个向量的长度为 H。伯特的位置嵌入是经过训练的，不像*那样是固定的，你所需要的只是注意力；*应用了一个丢弃，然后完成了图层标准化
5.  图层规范化参数= 1536 (2*H)。归一化需要学习两个参数——每个嵌入位置的平均值和标准偏差，因此是 2*H
6.  编码器:multi head self attention:multi head attention = 2362368

这需要一点解释。这是在这个步骤中发生的事情[ [ref](http://jalammar.github.io/illustrated-transformer/) ]:

![](img/c9fc924f3b7001a77eb1d65cc206d117.png)

共有 12 个头，输入尺寸为 768。因此每个头生成长度为 768/12 = 64 的嵌入。生成了三个嵌入— Q、K、v。总共:每个磁头 768*64*3 个参数，或者所有磁头 12*768*64*3 个参数。加上 Q，K，V 的偏差，还有 768*3。合计= 12 * 768 * 64 * 3+768 * 3；这是在连接所有头部之后。然后应用一个额外的权重(上图中向右的 W0)。那就是全连通的密集层，输出维度=输入维度。因此，参数(带偏差)= 768*768 + 768。所以这一步的总参数= A * D *(D/A)* 3+D * 3+D * D+D = 12 * 768 * 64 * 3+768 * 3+768 * 768+768 = 2362368

7.另一层标准化，遵循与#5 相同的逻辑

8.前馈:前馈。这实际上是一个前馈*网络*，它有*两个*完全连接的前馈层。它将输入维度(H)转换为 FFD，然后再转换回 H，其间有 ReLu 激活。所以有偏差的总参数=(H * FFD+FFD)+(FFD * H+H)=(768 * 3072+3072)+(3072 * 768+768)= 4722432；这跟随着另一个脱落层

9.另一层标准化，遵循与#5 相同的逻辑

步骤 6-9 涵盖了一个单一的变压器层，相同的设置重复 12(L)次

这遵循两个产出目标。一个用于 MLM(屏蔽语言建模)，一个用于 NSP(下一句预测)。让我们观察它们的参数:

10.MLM 密集:它接受一个嵌入作为输入，并试图预测屏蔽词的嵌入。所以参数(有偏差)= H * H + H = 768*768 + 768 = 590592

11.MLM-诺姆:标准化层，参数计数遵循与#5 相同的逻辑

12.MLM-Sim:嵌入相似性:这是计算 MLM 范数的输出和输入屏蔽令牌的嵌入之间的相似性。但是这一层也学习令牌级别偏差。这就是这一层中的 T (=30k)个参数(直觉上，我认为这类似于令牌级先验，但如果我错了，请纠正我)。

13.NSP 密集:密集:这将输入的 D 长度嵌入转换为另一个 D 长度嵌入。参数= D *D + D = 590592

14.NSP:密集:前一层的输出 D 长度嵌入然后被转换成两个向量，每个向量分别表示 IsNext 和 NotNext。因此，参数= 2*D + 2= 1538

整个网络的概述到此结束。通过这些，我得到了以下问题的答案:

1.  序列和位置嵌入来自哪里，以及两者都是可训练的事实
2.  详细了解变压器单元内部发生的情况
3.  图层规范化的作用
4.  同时传播 MLM 和 NSP 的任务

脚注:

[PyTorch 预排实施注意事项](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

[伯特中的三种嵌入类型](/@_init_/why-bert-has-3-embedding-layers-and-their-implementation-details-9c261108e28a)

[Colab 笔记本了解 BERT](https://colab.research.google.com/drive/1Nlhh2vwlQdKleNMqpmLDBsAwrv_7NnrB#forceEdit=true&sandboxMode=true&scrollTo=abSHyknGrI7b) 中的注意力——这也有一个很酷的交互可视化，解释了 Q，K，V 嵌入如何相互作用以产生注意力分布

[解释变形金刚、自我注意和交叉注意](http://jalammar.github.io/illustrated-transformer/)