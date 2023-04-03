# 如何使用 Ktrain 部署您的神经网络模型

> 原文：<https://medium.com/analytics-vidhya/how-to-deploy-your-neural-network-model-using-ktrain-ae255b134c77?source=collection_archive---------2----------------------->

![](img/50b2207a6119c1f789d44c53c3d9d738.png)

Mauro Sbicego 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**注意:**我假设你知道如何在 ktrain 中训练模型，但有一个基本的笔记本也显示了如何使用 ktrain，我已经在下面的库版本中运行了该笔记本。 ***你会在文章末尾找到文章源代码和参考资料。***

```
ktrain==0.26.4
tensorflow==2.5.0
```

**本文更新于 2021 年 7 月 9 日。**

当我在 [google colab](https://colab.research.google.com/) 中训练我的深度学习模型，做情感分析的时候。我发现了一篇有趣的文章 [***可解释的人工智能实践***](/@asmaiya/explainable-ai-in-practice-2e5ae2d16dc7) ，当我读到它时，我对它的酷感到震惊。我们如何在四行中训练我们的深度学习模型，各种方法用于文本分类，如 [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) ，这是一个最先进的类模型，fasttext 和其他许多技术都包含在 [**ktrain**](https://github.com/amaiya/ktrain) 中。ktrain 是 Keras 训练神经网络的轻量级包装器，它给了我非常令人惊讶的结果，也可以根据需要定制。所以，我们来做部署吧。

**第一步**:你必须使用 **learner** 变量训练你的模型，比如 **learner.autofit()** 和 **preproc** 用于预处理文本。所以，你要做的就是调用 ktrain 中的**get _ predict or**函数，使用 **save** 函数保存模型。

```
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('spam_text_message')
print('MODEL SAVED')
```

**第二步:**现在，我有了一个名为 **spam_text_message** 的文件夹，里面有两个文件 **tf_model.h5** 和 **tf_model.preproc.** 现在，我们可以加载模型进行预测了。

```
import pickle
from tensorflow.keras.models import load_model# loading preprocess and model file
features = pickle.load(open('spam_text_message/tf_model.preproc',
                            'rb'))
new_model = load_model('spam_text_message/tf_model.h5')
labels = ['ham', 'spam']
```

从上面的代码中，我们已经在 features 变量中加载了 **tf_model.preproc** 文件，在 new_model 变量中加载了 **tf_model.h5** ，并且**标签列** names 被 **one-hot** 编码。

第三步:现在，让我们做预测，但在此之前，我们必须预处理文本。所以，我们将使用**特性的**变量。

```
text = 'hey i am spam'
preproc_text = features.preprocess([text])
```

我已经在**特性**变量中调用了**预处理**函数，并将预处理文本保存在 **preproc_text** 变量中。

第四步:现在，我们可以做预测了。

```
result = new_model.predict(preproc_text)# OUTPUT =>array([[9.9999797e-01, 2.0015173e-06]], dtype=float32)
```

完成预测后，我们将得到**n 数组。**因此，我们将把数组转换成我们的标签，如果你愿意，你可以使用数组值作为分数。

**步骤 5:** 我有一个一次性编码的标签列表。

```
label = labels[result[0].argmax(axis=0)]
score = ('{:.2f}'.format(round(np.max(result[0]), 2)*100))
print('LABEL :', label, 'SCORE :', score)# OUTPUT => LABEL : ham SCORE : 100.00
```

如果您有任何疑问和建议，请随时提出，我会尽快解决。

谢谢，祝您愉快。

**参考资料:**关于 *ktrain、*的更多信息，请参见 *ktrain、* 上的 [**教程笔记本。**](https://github.com/amaiya/ktrain)

**源代码:**[kt rain-deployment-text-class ification . ipynb](https://github.com/ianuragbhatt/datascience-jupyter-notebooks/blob/master/ktrain/ktrain_deployment_text_classification.ipynb)