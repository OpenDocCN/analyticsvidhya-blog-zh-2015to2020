# 基于动作的视频分类(从头开始，无需 GPU 支持)

> 原文：<https://medium.com/analytics-vidhya/video-classification-based-on-action-without-gpu-f96ec9555197?source=collection_archive---------19----------------------->

没有 GPU！！没有外部重数据集！！阅读以学习和实现在任何机器上基于时间动作的基本视频分类技术。

在这里，我将创建自己的视频数据，其中，一个矩形在不同的方向移动。示例代码(使用 **Jupyter 笔记本**)如下:

```
import numpy as np
import skvideo.io as sk# creating sample video data
num_vids = 5
num_imgs = 100
img_size = 50
min_object_size = 1
max_object_size = 5

for i_vid in range(num_vids):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 vid_name = ‘vid’ + str(i_vid) + ‘.mp4’
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while x>0:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 x = x-1
 i_img = i_img+1
 sk.vwrite(vid_name, imgs.astype(np.uint8))# play a video
from IPython.display import Video
Video(“vid3.mp4”) # the script & video should be in same folder
```

现在我将创建 4 个不同类型的视频，其中，一个矩形在 4 个方向移动:左，右，上，下。因此，将有 4 个类，我将通过深度学习基于这些视频数据进行分类。浏览下面的代码(用 **Jupyter 笔记本**中的 **python 3.6.9，keras 2 . 2 . 4**)；肯定看评论。

```
import numpy as np**# preparing dataset**
X_train = []
Y_train = []
labels = enumerate([‘left’, ‘right’, ‘up’, ‘down’]) #4 classesnum_vids = 30
num_imgs = 30
img_size = 20
min_object_size = 1
max_object_size = 5# video frames with left moving object
for i_vid in range(num_vids):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 #vid_name = ‘vid’ + str(i_vid) + ‘.mp4’
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while x>0:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 x = x-1
 i_img = i_img+1
 X_train.append(imgs)
for i in range(0,num_imgs):
 Y_train.append(0)# video frames with right moving object
for i_vid in range(num_vids):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 #vid_name = ‘vid’ + str(i_vid) + ‘.mp4’
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while x<img_size:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 x = x+1
 i_img = i_img+1
 X_train.append(imgs)
for i in range(0,num_imgs):
 Y_train.append(1)# video frames with up moving object
for i_vid in range(num_vids):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 #vid_name = ‘vid’ + str(i_vid) + ‘.mp4’
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while y>0:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 y = y-1
 i_img = i_img+1
 X_train.append(imgs)
for i in range(0,num_imgs):
 Y_train.append(2)

# video frames with down moving object
for i_vid in range(num_vids):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 #vid_name = ‘vid’ + str(i_vid) + ‘.mp4’
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while y<img_size:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 y = y+1
 i_img = i_img+1
 X_train.append(imgs)
for i in range(0,num_imgs):
 Y_train.append(3)# data pre-processing
from keras.utils import np_utils
X_train=np.array(X_train, dtype=np.float32) /255
X_train=X_train.reshape(X_train.shape[0], num_imgs, img_size, img_size, 1)
print(X_train.shape)
Y_train=np.array(Y_train, dtype=np.uint8)
Y_train = Y_train.reshape(X_train.shape[0], 1)
print(Y_train.shape)
Y_train = np_utils.to_categorical(Y_train, 4)
```

> (120，30，20，20，1)
> (120，1)

```
**# building model**
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributedmodel = Sequential()
# TimeDistributed layer is to pass temporal information to the n/w
model.add(TimeDistributed(Conv2D(8, (3, 3), strides=(1, 1), activation=’relu’, padding=’same’), input_shape=(num_imgs, img_size, img_size, 1)))
model.add(TimeDistributed(Conv2D(8, (3,3), kernel_initializer=”he_normal”, activation=’relu’)))
model.add(TimeDistributed(MaxPooling2D((1, 1), strides=(1, 1))))
model.add(TimeDistributed(Flatten()))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False, dropout=0.3))
model.add(Dense(4, activation=’softmax’))
model.compile(optimizer=’adam’, loss=’categorical_crossentropy’, metrics=[‘accuracy’])
model.summary()**# model training**
model.fit(X_train, Y_train, nb_epoch=50, verbose=1)**# model testing with new data (4 videos)**
X_test=[]
Y_test=[]
for i_vid in range(2):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while x<img_size:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 x = x+1
 i_img = i_img+1
 X_test.append(imgs)
# 2nd class — ‘right’for i_vid in range(2):
 imgs = np.zeros((num_imgs, img_size, img_size)) # set background to 0
 w, h = np.random.randint(min_object_size, max_object_size, size=2)
 x = np.random.randint(0, img_size — w)
 y = np.random.randint(0, img_size — h)
 i_img = 0
 while y<img_size:
 imgs[i_img, y:y+h, x:x+w] = 255 # set rectangle as foreground
 y = y+1
 i_img = i_img+1
 X_test.append(imgs)
# 4th class — ‘down’X_test=np.array(X_test, dtype=np.float32) /255
X_test=X_test.reshape(X_test.shape[0], num_imgs, img_size, img_size, 1)pred=model.predict_classes(X_test)
pred
```

> 数组([1，1，3，3]，dtype=int64)

在这里，4 个测试视频被正确分类。

感谢阅读。也在这里浏览我的第一篇相关文章。