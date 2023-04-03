# 自定义 keras 生成器从 S3 获取图像以训练神经网络

> 原文：<https://medium.com/analytics-vidhya/custom-keras-generator-fetching-images-from-s3-to-train-neural-network-4e98694de8ee?source=collection_archive---------6----------------------->

![](img/d9a4fc564992aeabd994ccdedba66221.png)

赫克托·j·里瓦斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

AWS 中的训练模型可以通过许多不同的方式完成，使用任意数量的服务。今天，我想重点谈谈机器学习管道的一部分，你可以在你的模型上训练或预测。

我一直在开发一个概念验证模型，一旦我对最初的结果感到满意，下一步就是在云中扩展它，做一个更大规模的模型。我用 keras 框架中的模型进行了迁移学习，并训练了一个基于图像的模型。一个关键的 keras 函数是 predict_generator，它接受一个生成器作为输入。

长话短说，我一直使用 flow_from_directory 来读取图像，我认为它可能有可能改为在 S3 现场读取图像。一种方法是编写一个定制的 keras 生成器来读取图像并进行预处理。

在浪费您的时间之前，我应该正确地指出，从 S3 读取的延迟很高，即使对于小文件，每个文件的延迟也高达 0.5 秒，这被证明是一个大规模的瓶颈，可能会使解决方案变得毫无用处。在某些情况下，您可能能够忍受这种延迟。或者，压缩/捆绑文件也是一种解决方案。

以下是生成器的一些片段。让我们从第一个函数开始，它实际上是从 S3 获取数据。

```
from keras.preprocessing.image import load_img
import iodef fectch_input(path,s3): object = s3.Object(bucket_name,path)
     img = load_img(io.BytesIO(object.get()['Body'].read()))
     return(img)
```

s3 客户机作为输入传递。我使用的是 keras 加载模块，但是也可以使用通用的 PIL 包。接下来是一些简单的调整大小的例子，我再次使用 keras 处理:

```
from keras.preprocessing.image import img_to_arraydef preprocess_input(img): image = img.resize((128,128))
     array = img_to_array(image)
     return(array)
```

这里需要注意的一点是观察图像张量的格式，以确保你使用的是正确的格式。在这个项目中，我做的是迁移学习，匹配输入格式当然很重要。

```
def s3_image_generator(files, batch_size = 16):
     s3 = boto3.resource('s3') while True:
          #batch_paths = np.random.choice(a = files,
          #                               size = batch_size) batch_paths = np.array(files)
          batch_input = []
          batch_output = [0] * len(files) for input_path in batch_paths:
                input = fectch_input(input_path, s3) input = preprocess_input(input)
                batch_input += [ input ] batch_x = np.array( batch_input )
          batch_y = np.array( batch_output )
          yield( batch_x, batch_y )
```

我正计划进行无监督学习，所以我不需要标签，因此响应中有 0。如果您计划将此用于训练，可以使用 np.random.choice 进行随机采样。

所以基本上就是这样！下面是一个片段，用于实际传递 S3 路径，并使用生成器进行预测。

```
import pandas as pd
import boto3
from time import timebucket_name =''client = boto3.client('s3')
# Create a reusable Paginator
paginator = client.get_paginator('list_objects_v2')# Create a PageIterator from the Paginator
page_iterator = paginator.paginate(Bucket=bucket_name,Prefix =’’)g = pd.Series()a=time()
for page in page_iterator:
     m=pd.Series(list(map(lambda d: d['Key'], page['Contents'])))
     g = g.append(m)
     print(len(g))
     if len(g) == 10000:
          break
print(time() - a)g = g.reset_index(drop=True)preds = model_updated.predict_generator(s3_image_generator(g[0:512]), steps = 1, verbose = 1)
```

最后要注意的是，如果你只想做预测，这个流程可能有点过于复杂。没有生成器的替代方案是简单地加载和预测而没有生成器:

```
first = 1
a = time()
s3 = boto3.resource('s3')
for path in g[0:1000]:
    x_batch=[] object = s3.Object(bucket_name,path)
    img = load_img(io.BytesIO(object.get()['Body'].read()))
    image = img.resize((128,128))
    array = img_to_array(image) preds=(model.predict_on_batch(array))
    if first==1:
        predsA=preds.copy()
        first=0
     else:
        predsA=np.append(predsA,preds,axis=0)time()-a
```

感谢阅读。