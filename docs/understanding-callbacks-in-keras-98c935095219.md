# 了解 Keras 中的回调

> 原文：<https://medium.com/analytics-vidhya/understanding-callbacks-in-keras-98c935095219?source=collection_archive---------18----------------------->

![](img/c8106112a2f08c566ea6447c05954ed9.png)

照片由 Maria Casteli 在 Unsplash 上拍摄

训练深度学习模型是一个非常复杂的过程。在训练模型时，几乎不可能预测模型的这么多参数。例如，事先决定历元的数量是一项单调乏味的任务。在这篇文章中，我们将研究一些使用回调来控制模型训练的方法。

# 什么是回拨:

keras 中的回调有助于我们对模型进行适当的训练。从框架的角度来看，它是一个对象，我们可以在使用 fit 方法时将其传递给模型，并可以在训练的不同点调用它。下面我们可以看到在模型训练中可以使用的不同类型的回调。

## 提前停止:

当验证损失不再改善时，我们可以使用*提前停止*来中断训练过程。我们还可以确保在训练期间存储最佳模型。下面的代码片段显示了应用提前停止的方法。

```
keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=0,
                              mode='auto')
```

让我们逐个检查参数。

***监视器***——这个参数告诉我们应该被监视的性能指标。

***min_delta*** -该参数表示监控值的下限(*此处为 val_loss* )，我们可以将其视为改进。

***耐心-*** 该参数表示当我们的 val_loss 没有改善时，在停止训练过程之前我们可以等待的次数。

***模式-*** 它表示我们应该检查监控数量的方向(可能增加或减少)。

## 模型检查点:

它帮助我们在训练过程中的不同点保存模型的当前重量。下面的代码片段显示了实现的方法。

```
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath, monitor=’val_accuracy’, save_best_only=True)
```

***filepath*** -在每一个历元之后，将模型的权重保存在这个路径中。

***监控和保存 _ 最佳-*** 这两个参数确保保存最佳模型，除非模型没有根据监控值进行改进，否则我们不会保存结果。

## **高原减少:**

当验证损失停止改善时，我们可以使用这种回调来降低学习率。这对于在训练中走出局部极小点真的很有帮助。下面的代码片段显示了执行它的方法。

```
from keras.callbacks import ReduceLROnPlateaureduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5)
```

上面的代码片段说，如果 val_loss 在 5 个时期内没有改善，那么它会将学习率更改为其 1/5 的值。

为了将这些回调应用到我们的模型中，我们可以使用 fit 方法并传递回调列表。下面的代码片段显示了同样的情况。

```
model.fit(x,y,epochs=5,callbacks=callbacks_list,validation_data=(x_val,y_val)
```

这里 callback_list 可以是上面我们讨论过的任何一个回调。

这是一个在 keras 中训练模型的非常有效的机制的简要概述。