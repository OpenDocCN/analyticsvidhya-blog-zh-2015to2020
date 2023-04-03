# 处理 PyTorch 自定义数据集

> 原文：<https://medium.com/analytics-vidhya/dealing-with-pytorch-custom-datasets-64b6c40fe581?source=collection_archive---------5----------------------->

在本文中，我们将看看如何处理自定义 PyTorch 数据集。

![](img/f6f5237335b28a1a054d437b3e33d9b5.png)

[约书亚·厄尔](https://unsplash.com/@joshuaearle?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**自定义数据集！！为什么？？**

因为你可以用你渴望的方式塑造它！！！

在处理不同的项目时，我们很自然地会开发我们创建自定义数据集的方式。

PyTorch 上有一些官方的自定义数据集示例，比如这里的，但是对于初学者来说似乎有点晦涩难懂(就像我当时一样)。我们将要讨论的题目如下。

1.  **自定义数据集基础。**
2.  **使用 Torchvision 变换。**
3.  **对付熊猫(read_csv)**
4.  **将类嵌入到文件名中**
5.  **使用数据加载器**

# 1.自定义数据集基础。

数据集必须包含以下函数，供 DataLoader 稍后使用。

*   函数，初始逻辑在这里发生，如读取 CSV、分配转换、过滤数据等。,
*   `__getitem__()`返回数据和标签。
*   `__len__()`返回数据集的样本数。

现在，第一部分是创建一个数据集类:

```
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # stuff

    def __getitem__(self, index):
        # stuff
        return (img, label)

    def __len__(self):
        return count # of how many examples(images?) you have
```

这里，`MyCustomDataset`返回两件事，一个图像，和它的标签。但这并不意味着`__getitem__()`只限于归还那些。

**注:**

*   `__getitem()`返回单个数据点的特定类型(如张量)，否则，在加载数据时会出现错误，

```
TypeError: batch must contain tensors, numbers, dicts or lists; found <class 'PIL.PngImagePlugin.PngImageFile'>
```

# **2。使用火炬视觉变换:**

在大多数例子中，我们将在`__init__()`中看到`transforms = None`，它将为我们的数据/图像应用 Torchvision 变换。你可以在这里找到所有变换[的列表。](https://pytorch.org/docs/0.2.0/torchvision/transforms.html)

转换最常见的用法是:

```
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, ..., transforms=None):
        # stuff
        ...
        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        ...
        data = # Some data read from a file or image
        if self.transforms is not None:
            data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have

if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])
    # Call the dataset
    custom_dataset = MyCustomDataset(..., transformations)
```

您可以在数据集类中定义转换。像这样:

```
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # stuff
        ...
        # (2) One way to do it is define transforms individually
        self.center_crop = transforms.CenterCrop(100)
        self.to_tensor = transforms.ToTensor()

        # (3) Or you can still compose them like 
        self.transformations = \
            transforms.Compose([transforms.CenterCrop(100),
                                transforms.ToTensor()])

    def __getitem__(self, index):
        # stuff
        ...
        data = # Some data read from a file or image

# When you call transform for the second time it calls __call__()               and applies the transform         data = self.center_crop(data)  # (2)
        data = self.to_tensor(data)  # (2)

        # Or you can call the composed version
        data = self.transformations(data)  # (3)

     # Note that you only need one of the implementations,(2) or (3)
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have

if __name__ == '__main__':
    # Call the dataset
    custom_dataset = MyCustomDataset(...)
```

# **3。对付熊猫(read_csv):**

现在我们的数据集包含一个文件名、标签和一个额外的操作指示器，我们将对图像执行一些额外的操作。

```
+-----------+-------+-----------------+
| File Name | Label | Extra Operation |
+-----------+-------+-----------------+
| tr_0.png  |     5 | TRUE            |
| tr_1.png  |     0 | FALSE           |
| tr_1.png  |     4 | FALSE           |
+-----------+-------+-----------------+
```

构建从该 CSV 读取图像位置的自定义数据集。

```
class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """# Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Third column is for an operation indicator
        self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # Check if there is an operation
        some_operation = self.operation_arr[index] # If there is an operation
        if some_operation:
            # Do some operation on image
            # ...
            # ...
            pass # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)

        # Get label of the image based on the cropped pandas column single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    # Call dataset
    custom_mnist_from_images =  \
        CustomDatasetFromImages('../data/mnist_labels.csv')
```

另一个从 CSV 读取图像的例子，其中每个像素的值都列在列中(例如， **MNIST** )。`__getitem__()`中逻辑的一点变化。最后，我们只是将图像作为张量和它们的标签返回。数据看起来像这样，

```
+-------+---------+---------+-----+
| Label | pixel_1 | pixel_2 | ... |
+-------+---------+---------+-----+
|     1 |      50 |      99 | ... |
|     0 |      21 |     223 | ... |
|     9 |      44 |     112 |     |
+-------+---------+---------+-----+
```

现在，代码看起来像这样:

```
class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """ self.data = pd.read_csv(csv_path)
        self.labels = np.asarray(self.data.iloc[:, 0])
        self.height = height
        self.width = width
        self.transforms = transform

    def __getitem__(self, index):
        single_image_label = self.labels[index] # Read each 784 pixels and reshape the 1D array ([784]) to 2D array ([28,28]) 
        img_as_np = np.asarray(self.data.iloc[index][1:]).reshape(28,28).astype('uint8') # Convert image from numpy array to PIL image, mode 'L' is for grayscale
        img_as_img = Image.fromarray(img_as_np)
        img_as_img = img_as_img.convert('L') # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img) # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)

if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv', 28, 28, transformations)
```

# **4。将类名嵌入为文件名:**

使用图像的文件夹名作为文件名:

```
class CustomDatasetFromFile(Dataset):
    def __init__(self, folder_path):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms

        Args:
            folder_path (string): path to image folder
        """ # Get image list
        self.image_list = glob.glob(folder_path+'*') # Calculate len
        self.data_len = len(self.image_list)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index] # Open image
        im_as_im = Image.open(single_image_path)

        # Do some operations on image
        # Convert to numpy, dim = 28x28
        im_as_np = np.asarray(im_as_im)/255 # Add channel dimension, dim = 1x28x28
        # Note: You do not need to do this if you are reading RGB images
        # or i there is already channel dimension im_as_np = np.expand_dims(im_as_np, 0) # Some preprocessing operations on numpy array
        # ...
        # ...
        # ...

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()

        # Get label(class) of the image based on the file name
        class_indicator_location = single_image_path.rfind('_c') label = int(single_image_path[class_indicator_location+2:class_indicator_location+3]) return (im_as_ten, label)

    def __len__(self):
        return self.data_len
```

# **5。使用数据加载器:**

PyTorch 数据加载器将调用`__getitem__()`并将它们打包成一个批处理。但是从技术上来说，我们不会使用数据加载器，一次调用一个`__getitem__()`，并将数据输入模型。现在，我们可以像这样调用数据加载器:

```
...
if __name__ == "__main__":
    # Define transforms
    transformations = transforms.Compose([transforms.ToTensor()]) # Define custom dataset
    custom_mnist_from_csv = \
        CustomDatasetFromCSV('../data/mnist_in_csv.csv',
                             28, 28,
                             transformations) # Define data loader
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv,
                                                    batch_size=10,
                                                    shuffle=False)

    for images, labels in mn_dataset_loader:
        # Feed the data to the model
```

这里，batch_size 决定了在一个批处理中将包装多少个单独的数据点。数据加载器将返回一个形状张量(批次—深度—高度—宽度)

```
tensor.shape(10x1x28x28) # if batch_size =10 (For MNIST Data). 
```

就是这样！！！

自定义数据集！！别担心！！

**参考:**

*   [https://github . com/utkuozbulak/py torch-custom-dataset-examples # a-custom-custom-custom-dataset](https://github.com/utkuozbulak/pytorch-custom-dataset-examples#a-custom-custom-custom-dataset)