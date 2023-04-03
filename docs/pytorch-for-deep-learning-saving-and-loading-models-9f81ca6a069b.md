# 用于深度学习的 PyTorch 保存和加载模型

> 原文：<https://medium.com/analytics-vidhya/pytorch-for-deep-learning-saving-and-loading-models-9f81ca6a069b?source=collection_archive---------4----------------------->

![](img/d9ec596172fd4321ec5014c3d4b302ea.png)

在本帖中，我们将看到如何在 pytorch 中保存和加载模型

# 为什么要保存模型？

尽管正在使用框架，保存模型是一件非常重要的事情，以便随时再次使用它，而不是从头开始再次训练神经网络。

有时，训练甚至可能需要数周才能完成。所以，每当我们需要使用我们的网络时，我们不能从一开始就训练他们。

当场保存模型的另一个好处是，我们可以比较两种不同的模型，选择哪一种对给定的任务更有效。

说到这里，让我们直接进入代码

1.  **导入一些库**

```
#importing the libraires
import torch
import torch.nn as nn
```

**2。创建虚拟线性网络**

```
#creating a linear modelclass Model(nn.Module):
  def __init__(self):
    super(Model,self).__init__()
    self.fc1 = nn.Linear(in_features=3,out_features=5) def forward(self,x):
    return self.fc1(x)
```

**3。保存模型**

```
#saving the model
model = Model()
torch.save(model,'something.h5')
```

torch.save 是一个接受两个参数的函数。
一个是模型本身。
第二个是需要保存模型的文件的路径。可以使用 *.h5* 文件格式或*。pth* 文件格式

**4。加载模型**

```
#loading the modelloaded_model = torch.load('something.h5')
```

torch.load 是一个函数，可用于将模型加载回变量中。
它采用的参数是保存原始模型的文件的路径，并返回可以存储在 python 变量中的模型

**5。将原始模型的权重与保存的模型进行比较**

```
#weights of the original modelprint(model.fc1.weight)**output:** Parameter containing: 
tensor([[ 0.0725,  0.1615, -0.0047],
        [-0.0371, -0.2640, -0.3004],         
        [ 0.2129,  0.1725, -0.0136],         
        [-0.5025,  0.5496,  0.0448],         
        [-0.2974,  0.1040,  0.0932]], requires_grad=True)#weights of the loaded modelprint(loaded_model.fc1.weight)**output:**Parameter containing: 
tensor([[ 0.0725,  0.1615, -0.0047],
        [-0.0371, -0.2640, -0.3004],         
        [ 0.2129,  0.1725, -0.0136],         
        [-0.5025,  0.5496,  0.0448],         
        [-0.2974,  0.1040,  0.0932]], requires_grad=True)
```

**6。使用加载的模型进行预测**

```
#making predictions with a loaded modeltensor = torch.tensor([[1,2,3]],dtype=torch.float32)
loaded_model(tensor)**output:** tensor([[ 0.7429, -1.2462,  0.4356,  1.0646,  0.5010]],        grad_fn=<AddmmBackward>)
```

# 结论

![](img/711857401a9e526247a27dde4c52c5e0.png)

正如我们已经看到的，保存模型非常重要，尤其是当涉及深度神经网络时。因此，在大型项目中工作时，不要忘记保存模型。

上面的代码是在 pytorch 中保存模型的简单方法。还有其他方法可以做到这一点。也许，那些会在其他博客文章中涉及。

代码文件可以在我的 github 存储库中找到

# 谢谢你