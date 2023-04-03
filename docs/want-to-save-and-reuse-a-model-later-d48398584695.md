# 想要保存并在以后重用模型吗？

> 原文：<https://medium.com/analytics-vidhya/want-to-save-and-reuse-a-model-later-d48398584695?source=collection_archive---------11----------------------->

![](img/7235b4568c9a41e8f6c4956beea0236a.png)

Olia Gozha 在 Unsplash 上拍摄的照片

在机器学习中，训练一个模型，测试它，绝对不是目的。我们应该运行这个训练的源代码，再次调整一切来做未来的预测吗？不需要！！！

有一些很酷的方法可以保存模型，并在以后加载它以供使用。

有一些包可以将你的机器学习模型保存到文件系统中，并在你想要重用它的时候加载它！。

> 使用 Pickle 包
> 
> 从 sklearn.externals 使用 joblib

# **操作步骤:**

1.  将模型保存为文件
2.  从保存的文件中加载模型
3.  使用加载的模型进行预测

# **使用 Pickle 将其保存为文件:**

导入所需的包

```
**import** **pickle** **as** **pkl**
```

你可以在注释中找到解释

```
filenm = 'LR_AdmissionPrediction.pickle'
***#Step 1: Create or open a file with write-binary mode and save the model to it***
pickle = pkl.dump(lr, open(filenm, 'wb'))***#Step 2: Open the saved file with read-binary mode***
lr_pickle = pkl.load(open(filenm, 'rb'))***#Step 3: Use the loaded model to make predictions*** 
lr_pickle.predict([[300,85,5,5,5,8,1]])
```

输出:

```
array([0.61745881])
```

在上面的代码中，我们创建/打开了一个名为“LR_AdmissionPrediction.pickle”的文件。

您可以在您的文件系统中找到这个文件。这将是一个非常小的文件。

大小:此文件保存模型，而不是用于训练或测试模型的数据。因此，该文件的大小并不反映用于训练/测试模型的数据的大小。

wb，rb:我们用写二进制打开文件来创建 pickle 文件，然后用读二进制加载它。

这里，我们已经序列化了我们的模型，并将其保存在文件系统中。

要知道 lr 从何而来，请查看[线性回归——第四部分——录取几率预测](/analytics-vidhya/linear-regression-part-iv-chance-of-admission-prediction-978540555c29)。

这不是唯一的方法。还有一种方法可以让你把它保存在一个字符串中，并把它加载到同一个项目中字符串范围可用的任何地方。下面是源代码。

```
***#Step 1: Save the model as a pickle string.***
saved_model = pkl.dumps(lr) 

***#Step 2: Load the saved model*** 
lr_from_pickle = pkl.loads(saved_model) 

***#Step 3: Use the loaded model to make predictions*** 
lr_from_pickle.predict([[300,85,5,5,5,8,1]])
```

输出:

```
array([0.61745881])
```

# 使用 joblib 将其保存为文件:

```
**from** **sklearn.externals** **import** **joblib** 

*#****Step 1:*** *Save the model as a pickle in a file* 
joblib.dump(lr, 'filename.pkl') 

*#****Step 2:*** *Load the model from the file* 
lr_from_joblib = joblib.load('filename.pkl')  

*#****Step 3:*** *Use the loaded model to make predictions* 
lr_from_joblib.predict([[300,85,5,5,5,8,1]])
```

输出:

```
array([0.61745881])
```

# **结论:**

在以上三种方法中，对于相同的输入，预测值是相同的。这意味着您可以使用任何方法来重用模型。

如果你发现任何其他类似的很酷的东西，或者如果你发现任何更正，我真的很感激，请在评论中添加它。

谢谢大家！

喜欢支持？只要点击拍手图标就可以了。

编程快乐！