# 图像分类:猫和狗——预训练神经网络与构建神经网络

> 原文：<https://medium.com/analytics-vidhya/image-classification-cats-and-dogs-pre-trained-neural-network-vs-constructed-6370d5c79fde?source=collection_archive---------8----------------------->

![](img/89ec8d0007ed555658badeda172e173b.png)

Python 神经网络项目的第 4 章介绍了一个从微软提供的数据集中对猫和狗进行分类的指导性项目。在我看来，对图像进行分类的最佳方式是使用卷积神经网络(CNN)。

我已经使用 VGG-16 CNN 对澳大利亚的金矿进行了分类，这是我在熨斗学校沉浸式数据科学训练营的压轴戏项目。CNN 可以用于各种图像分类和分割问题。

这个场景是一个非常基本的分类，二进制，它甚至不需要 GPU。要用 CNN 运行更复杂的图像分类问题，推荐使用好的 GPU。对多个高分辨率图像进行分类将需要比 CPU 大得多的计算能力。Kaggle 是利用云计算和访问 GPU 的一种很好的方式，但是，它有 30 小时/周的限制。

我碰巧能够用我的笔记本电脑 CPU 在大约一个半小时内为这个数据集训练一个 CNN。我还使用了一个预训练的模型，VGG16，以缩短学习/训练时间并比较结果。

以下是第一个模型的代码:

```
**from** **keras.models** **import** Sequential
**from** **keras.layers** **import** Conv2D, MaxPooling2D
**from** **keras.layers** **import** Dropout, Flatten, Dense
**from** **keras.preprocessing.image** **import** ImageDataGeneratormodel = Sequential()
```

在[47]中:

```
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10
```

在[21]中:

```
model.add(Conv2D(NUM_FILTERS,(FILTER_SIZE,FILTER_SIZE), input_shape = (INPUT_SIZE,INPUT_SIZE,3), activation = 'relu'))
```

在[22]中:

```
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE,MAXPOOL_SIZE)))
```

在[23]中:

```
model.add(Conv2D(NUM_FILTERS,(FILTER_SIZE,FILTER_SIZE), input_shape = (INPUT_SIZE,INPUT_SIZE,3), activation = 'relu'))
```

在[24]中:

```
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE,MAXPOOL_SIZE)))
```

在[25]中:

```
model.add(Flatten())
```

在[26]中:

```
model.add(Dense(units=128,activation ='relu'))
```

在[27]中:

```
model.add(Dropout(0.5))
```

在[28]中:

```
model.add(Dense(units=1,activation = 'sigmoid'))
```

在[29]中:

```
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
```

在[30]中:

```
training_data_generator = ImageDataGenerator(rescale = 1./255)training_set = training_data_generator.flow_from_directory('Dataset/PetImages/Train', target_size = (INPUT_SIZE,INPUT_SIZE),
                                                          batch_size = BATCH_SIZE,class_mode='binary')
model.fit_generator(training_set,steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose = 1)Found 19997 images belonging to 2 classes.
WARNING:tensorflow:From C:\Users\mmsub\Anaconda3\envs\learn-env\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.cast instead.
Epoch 1/10
1250/1250 [==============================] - 80s 64ms/step - loss: 0.6250 - accuracy: 0.6488
Epoch 2/10
1250/1250 [==============================] - 76s 61ms/step - loss: 0.5393 - accuracy: 0.7292
Epoch 3/10
1250/1250 [==============================] - 80s 64ms/step - loss: 0.4925 - accuracy: 0.7648
Epoch 4/10
1250/1250 [==============================] - 84s 67ms/step - loss: 0.4660 - accuracy: 0.7787
Epoch 5/10
1250/1250 [==============================] - 76s 61ms/step - loss: 0.4353 - accuracy: 0.7956
Epoch 6/10
1250/1250 [==============================] - 76s 61ms/step - loss: 0.4123 - accuracy: 0.8112
Epoch 7/10
1250/1250 [==============================] - 86s 69ms/step - loss: 0.3876 - accuracy: 0.8257
Epoch 8/10
1250/1250 [==============================] - 84s 67ms/step - loss: 0.3650 - accuracy: 0.8354
Epoch 9/10
1250/1250 [==============================] - 76s 61ms/step - loss: 0.3448 - accuracy: 0.84520s - loss: 0.3
Epoch 10/10
1250/1250 [==============================] - 78s 62ms/step - loss: 0.3277 - accuracy: 0.8557
```

在[33]中:

```
testing_data_generator = ImageDataGenerator(rescale = 1./255)test_set = testing_data_generator.flow_from_directory('Dataset/PetImages/Test/', target_size = (INPUT_SIZE,INPUT_SIZE),
                                                     batch_size =BATCH_SIZE, class_mode = 'binary')
score = model.evaluate_generator(test_set,steps =len(test_set))
**for** idx, metric **in** enumerate(model.metrics_names):
    print("**{}**: **{}**".format(metric,score[idx]))
```

OUT[33]:

```
Found 5000 images belonging to 2 classes.
loss: 0.4586998224258423
accuracy: 0.7900000214576721
```

我构建的模型有相当不错的 79%的准确率，但也花了一个半小时来训练。用 VGG16 预先训练的第二个模型花费 27 分钟来训练，并且具有 87%的更好的准确度分数。

预训练的 VGG16 型号代码:

```
**from** **keras.applications.vgg16** **import** VGG16
```

在[71]:

```
INPUT_SIZE = 128
vgg16 = VGG16(include_top = **False**, weights = 'imagenet',input_shape=(INPUT_SIZE,INPUT_SIZE,3))
```

在[72]中:

```
**for** layer **in** vgg16.layers:
    layer.trainable=**False**
```

在[73]:

```
**from** **keras.models** **import** Model

input_ = vgg16.input
output_=vgg16(input_)
last_layer = Flatten(name='flatten')(output_)
last_layer = Dense(1,activation ='sigmoid')(last_layer)
model = Model(input=input_, output = last_layer)
```

在[74]:

```
BATCH_SIZE = 16
STEPS_PER_EPOCH = 200
EPOCHS = 3
```

在[75]:

```
model.compile(optimizer ='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
```

在[77]:

```
training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)

training_set = training_data_generator.flow_from_directory('Dataset/PetImages/Train/', target_size=(INPUT_SIZE,INPUT_SIZE),
                                                           batch_size = BATCH_SIZE, class_mode = 'binary')
test_set = testing_data_generator.flow_from_directory('Dataset/PetImages/Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = BATCH_SIZE,
                                             class_mode = 'binary')
model.fit_generator(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs = EPOCHS, verbose =1)Found 19997 images belonging to 2 classes.
Found 5000 images belonging to 2 classes.
Epoch 1/3
200/200 [==============================] - 560s 3s/step - loss: 0.4012 - accuracy: 0.8041
Epoch 2/3
200/200 [==============================] - 575s 3s/step - loss: 0.2994 - accuracy: 0.8711
Epoch 3/3
200/200 [==============================] - 488s 2s/step - loss: 0.2669 - accuracy: 0.8894
```

Out[77]:

```
<keras.callbacks.callbacks.History at 0x1e18ad49ba8>
```

在[78]:

```
score = model.evaluate_generator(test_set,len(test_set))

**for** idx, metric **in** enumerate(model.metrics_names):
    print("**{}**: **{}**".format(metric,score[idx]))loss: 0.6551424264907837
accuracy: 0.8781999945640564
```

正如我们所见，使用预先训练的模型比从头开始构建 CNN 要快得多，也更准确。使用预先训练好的模型的好处是，大部分工作已经完成，我们可以添加到它们上面。它们还需要更少的训练时间，这对大型数据集很重要。我的这个项目的完整代码可以在这里找到。