# [tensor flow 2.0]保存和恢复模型

> 原文：<https://medium.com/analytics-vidhya/tensorflow-2-0-save-and-restore-models-4708ed3f0d8?source=collection_archive---------2----------------------->

为了更好地理解，将会对某些部分进行修改和阐述，但是，我在此承认以下帖子是基于中提供的 TensorFlow 教程:

 [## 保存和恢复模型| TensorFlow 核心

### 您看到了如何将权重加载到模型中。手动保存它们与模型一样简单

www.tensorflow.org](https://www.tensorflow.org/beta/tutorials/keras/save_and_restore_models) 

有关代码和数据集的更详细的解释和背景知识，您可以随时查阅链接。

# 一.导言

## 我们为什么要保存模型？

我们需要一些动机来开始这样一个保存和恢复模型的漫长过程，那么我们到底为什么要做这样一个负担呢？

目的很简单，为什么我们要保存一个简单的 word 文件或在两个小时的讲座中做的笔记？然而，我们保存一个模型所能达到的效率是保存几页长的 word 文件所不能比拟的。

创建的模型可以在训练过程中和训练后保存。即使这个中等的草稿也能保存每秒发生的任何变化，为什么不是张量流模型呢？这个过程是通过*在训练期间保存检查点来完成的。*在检查点和回调**的帮助下，你可以在训练期间和结束时持续保存模型。**当 medium 自动保存您的草稿或您的帖子在 Instagram 上“临时保存”时，我们感到安全，TensorFlow 检查点也是如此；你现在不会受到工作中可能出现的任何干扰。

## 我们想拯救什么？

我们可以只保存权重来恢复它们或整个模型，这样您就不必重新进行整个训练。保存的最大好处是，您可以将您的模型共享给其他人，以便他们可以参考您的代码，或者在您的模型的训练结果的基础上，以更高的准确性和效率创建他们的模型。

## 我们如何“恢复”保存的模型？

一旦你保存了你的微软 Word 文件，你只需要双击它或者你喜欢的任何方式打开它。此外，我们经常打开一个下载的文件，并从那里开始你的工作，或者只是参考你的参考文件。同样的事情也发生在模型上，但是比双击桌面上的文件名稍微复杂一些。

# 二。设置

**1 —安装或导入 Tensorflow**

```
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass!pip install -q pyyaml h5py  # Required to save models in HDF5 formatfrom __future__ import absolute_import, division, print_function, unicode_literalsimport osimport tensorflow as tf
from tensorflow import kerasprint(tf.version.VERSION)
```

**正如带有 hashtag 的代码中所说，您应该在 Colab 中练习代码**

[https://colab.research.google.com](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true)

> (边注)我刚不小心按了 command+w 回来！！对于 Mac 用户，你会看到刚刚发生在我身上的事情，但是自动保存确实救了我！我还需要 30 分钟来重写这一切，或者花 3 个小时来写一个 tensorflow 代码！所以，请学习如何保存你的模型。；)

**2 —导入数据集**

*   在本练习中，我们使用 MNIST 数据集和前 1000 个示例。

```
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()train_labels = train_labels[:1000]
test_labels = test_labels[:1000]train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```

**3 —定义模型**

*   我们将建立一个简单的序列模型。

```
# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])return model
```

*   这里定义的 create_model()'函数将不断地被用来使代码更简单。
*   所以用' model=create_model()'，我们就完成了模型的构建。

```
# Create a basic model instance
model = create_model()# Display the model's architecture
model.summary()
```

*   **model.summary( )** 的结果如下:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

我被教导说，我不应该认为数字是理所当然的。现在，我想把这个习惯传递给可能在网上的读者。

*   **input_shape=(784，)**来自 **reshape(-1，28*28):** 在建模之前，我们将 28*28 的二维图像文件整形为长度为 28*28=784 的 1D 矢量
*   **第一个密集层的参数# 401920** :该层有(784+1)*512 个参数。
    ***keras . layers . dense(512，activation='relu '，input_shape=(784，)***
    512 是第一个密集层的节点数(⁹).
    784 是对应于该层作为输入的 1D 向量的权重数。
    1 是将要添加的偏差数。
    所以每个节点都以 W1*X1+…+W784*X784 + B1 作为输入！
*   **keras . layers . dropout(0.2)**通过将输入单元的某个分数(0.2)随机设置为零，来防止通过历元的过度拟合。我把它理解为放松身体肌肉，深呼吸以做出更好的瑜伽姿势。通过将负担(数据)保持在您可以放心管理的水平，我们有时可以实现更多。
*   **参数#5130** 来自第二密集层:在这一层中，有(512+1)*10 个参数。 ***keras . layers . dense(10，activation = ' soft max ')***
    10 是第二密集层的节点数。
    512 是对应于输入长度或第一层节点数的权重数。
    1 是偏置数。

> 旁注)参考以下教程以了解有关 MNIST 数据集或建模过程的更多信息[https://www . tensor flow . org/beta/tutorials/keras/basic _ classification](https://www.tensorflow.org/beta/tutorials/keras/basic_classification)

# 三。在训练期间保存检查点，并将其恢复到新的未训练模型

通过保存检查点，如果建模由于任何可能的动机而中断，人们可以使用经过训练的模型，而无需重做整个过程或从中间重新开始。

**1。仅在训练期间保存重量**

```
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)**# Create a callback that saves the model's weights** cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                   **save_weights_only=True**,
                                   verbose=1)**# Train the model with the new callback** model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training**# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.**
```

该过程将留下在每个时期结束时更新的 Tensorflow 检查点文件的单一集合。

我们可以通过以下方式看到这一点:

```
!ls {checkpoint_dir}
```

其结果是:

```
checkpoint           cp.ckpt.data-00001-of-00002
cp.ckpt.data-00000-of-00002  cp.ckpt.index
```

**2。从纯权重模型中恢复一个模型，并将其应用于一个新的未经训练的模型**

不幸的是，与 word 文件或中型草稿不同，检查点**不会**保存模型的所有内容。其实更多的是你看完长篇论文后做的旁注。所以，现在您想打开保存的便笺并创建您的另一张便笺。

当从权重恢复一个模型时，你总是需要一个与原始模型结构完全相同的模型。 ***一旦你有了相同的模型架构，你就可以共享权重，尽管它是一个模型的不同实例。***

你不能用专门为 Galaxy Note 10 设计的充电器给你的全新 iPhone 11 充电。然而，尽管手机的功能不同，你仍然可以用你用来给 iPhone XR 充电的充电器给 iPhone 11 充电。(除非他们改变了标准)

*   现在，重新构建一个新的未训练模型，然后在测试集上对其进行评估。

```
# Create a basic model instance
model = create_model()# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
```

这将导致相当令人沮丧的精度水平。(很明显，您没有接受过培训——如果模型正确地将图像文件与正确的标签相匹配，这仅仅是一种运气)

*   现在，加载我们之前在 checkpoint 中保存的重量，并重新评估模型。

```
# Loads the weights
model.load_weights(checkpoint_path)# Re-evaluate the model
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

将精度与我们在原始模型中得到的精度进行比较。

下面，将其与以下模型拟合之前获得的结果进行比较:

```
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])
```

**3。检查点回调选项**

正如您可能已经预料到的那样，总是有修改一些关于检查点的选项的空间。

比如看论文或者备考的时候有必要一行一行做笔记吗？对于一篇极其庞大的论文来说，主题或概念在整篇论文中不会改变，为了提高效率，你可能需要跳过一些页面。同样，回调提供了一个选项，以便您可以在每个特定时期保存一次检查点。

此外，为了方便起见，您可以为检查点命名。

现在，让我们训练一个新模型，每五个历元保存一次不同名称的检查点。

**3.1。定义检查点路径和检查点目录**

```
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = **"training_2/cp-{epoch:04d}.ckpt"** checkpoint_dir = os.path.dirname(checkpoint_path)**#previoulsy, checkpoint_path = "training_1/cp.ckpt"**
```

**3.2。创建一个保存模型权重的回调**

```
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    **period=5**)**# previously, there was no period defined** 
```

**在第 3.3 和 3.4 部分中，我们将在保存时进行训练，然后创建一个新模型并重新加载保存的权重。与我们对基本检查点所做的过程完全相同(不带选项的版本)**

**3.3 用新的回调(训练)创建新的模型和模型拟合**

```
# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
**model.save_weights(checkpoint_path.format(epoch=0))** 
# Train the model with the new callback
model.fit(train_images, 
          train_labels,
         ** epochs=50**, 
          callbacks=[cp_callback],
          validation_data=(test_images,test_labels),
          verbose=0)
```

然后，我们希望查看生成的检查点列表，然后检查最新的检查点，并将其命名为“latest”

```
! ls {checkpoint_dir}latest = tf.train.latest_checkpoint(checkpoint_dir)
latest
```

这将导致

```
'training_2/cp-0050.ckpt'
```

**3.4。重新创建一个模型并重新加载最新的检查点**

```
# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

准确度的结果如何不同或相似？

**4。手动保存重量**

```
# Save the weights
model.save_weights(**'./checkpoints/my_checkpoint'**)
**#previously, *model.save_weights(checkpoint_path.format(epoch=0))*** # Create a new model instance
model = create_model()

# Restore the weights
model.load_weights(**'./checkpoints/my_checkpoint'**)
***#previously, model.load_weights(checkpoint_path)*** # Evaluate the model
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

# **四。保存整个模型**

如果要整体保存模型呢？检查点方法只保存权重，但是通过将整个模型保存到一个文件中，我们可以共享一个模型并重新加载它，而无需重新定义一个应该与之前完全相同的模型。

在我们的例子中，我们定义了 *'create_model()'* ，所以我们确信任何新创建的带有 *'model= create_model()'* 的模型实例都将具有完全相同的模型架构。但是，如果你想重新加载在网上下载的模型呢？如果你没有下载代码的完整信息，可能会有些困难。

因此，通过保存一个完整的模型是非常有用的，从某种意义上说，你总是可以从保存的模型完成其过程的部分开始，并运行你自己的代码或模型。

1.  **将模型保存为 HDF5 文件**

*   将模型保存到 HDF5 文件中相当简单；***' model . save(' name . H5 ')'***

```
# Create a new model instance
model = create_model()

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file
model.save('my_model.h5')
```

**2。从文件**重新创建模型

*   更多的是从您或其他作者离开的部分重新加载模型。

```
# Recreate the exact same model, including its weights and the optimizer
new_model = keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()
```

结果和以前完全一样；(检查二。设置，第 3 点。)

```
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_10 (Dense)             (None, 512)               401920    
_________________________________________________________________
dropout_5 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
_________________________________________________________________
```

我们可以重新评估模型的准确性；

```
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
```

因此，通过保存模型，可以保存权重、模型架构和优化器。

希望这篇文章对你理解保存和重新加载 tensorflow 模型有所帮助。我试图根据自己的类比来总结和解释这些概念，但我希望这不会让人们更加困惑。

这里没有人是专家，所以可能会有一些错误。因此，任何抬头是完全欢迎的，我将很荣幸有人问我问题！

祝你有愉快的一天！