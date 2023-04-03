# 使用 Flask 创建您的第一个 ML Web 应用程序

> 原文：<https://medium.com/analytics-vidhya/create-your-first-ml-web-app-with-flask-ed0c4bb54312?source=collection_archive---------9----------------------->

![](img/e1c089540a48f3d1cef2bee300fd9249.png)

创建机器学习或深度学习模型很棒，但这种模型的部署将更棒。如果你在这里，你已经知道一些关于机器学习和模型构建的知识。模型部署部分是数据科学生命周期中最重要的部分之一。因此，我们将创建一个迷你 web 应用程序，以了解如何将您的 ml 模型转换为 web 应用程序。Web 应用程序提供了一个更棒的外观来与您的模型进行交互。

在这里，我将采用整个机器学习中最流行的数据集，你猜对了，我们将在 **IRIS** 数据集上工作。我把一个玩具数据集作为这里的主要目标是理解我们如何用一个简单的 ml 模型创建一个 web 应用程序。

让我们在 iris 数据集上快速创建我们的模型

**导入重要库**

```
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
```

Pickle —保存我们将要训练的模型，以便我们可以在 web 应用程序中直接使用它。

**数据**

```
data=pd.read_csv('iris.csv')
# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]# y = target values, last column of the data frame
y = data.iloc[:, -1]#Split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
```

现在，我们已经加载了数据，并将其分为测试和训练。

**创建并保存模型**

```
#Train the model
model = LogisticRegression()
model.fit(x_train, y_train) #Training the model
#Test the model
predictions = model.predict(x_test)
print( classification_report(y_test, predictions) )print( accuracy_score(y_test, predictions))
```

输出:

```
precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       1.00      1.00      1.00        11

       accuracy                           1.00        30
      macro avg       1.00      1.00      1.00        30
   weighted avg       1.00      1.00      1.00        30

1.0
```

现在我们终于训练好了我们的模型，我们有 100 %的准确率，所以现在让我们保存我们的模型，并在我们的 web 应用程序中使用它。

**酸洗我们的模型**

```
pickle.dump(model,open('model.pkl','wb'))
```

我们已经用 moel.pkl 名称保存了我们的模型。让我们快速地看看如何预测一个单一的身份。

```
p=model.predict([[5.1,3.5,1.4,0.2]])
print(p[0])'Iris-setosa'   # Output
```

我们刚刚通过以数组的形式传递所有特性做出了我们的预测，并且我们成功地得到了我们的结果，现在让我们开始构建我们的 Web 应用程序。

你可以在这里查看实现的网站演示。

[https://floweriris.herokuapp.com/](https://floweriris.herokuapp.com/)

# Web 应用程序

让我们开始创建我们的 web 应用程序。首先，创建一个文件名 **app.py。**现在让我们从开始编码。

![](img/c81d13e8ee961047f8cfdb522353893e.png)

杰弗逊·桑托斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

**导入库**

```
import numpy as npfrom flask import Flask, request, jsonify, render_templateimport pickle
```

现在，让我们使用 pickle 命令加载已经保存的模型。

```
model = pickle.load(open('model.pkl', 'rb'))
```

请确保您将模型保存在同一目录中，或者尝试提供模型的完整路径。

现在让我们开始创建我们的应用程序

```
app = Flask(__name__)@app.route('/')def home(): return render_template('index.html')
```

@app.route:表示当浏览器点击特定的 URL 时要做什么，为此，我们在它下面编写函数。在这种情况下，当用户试图打开我们的网页，它将呈现**index.html。**

现在让我们写我们的函数来做预测，并打印在我们的网页上。

让我们来理解一下上面的代码，首先我们写了一个函数，当我们浏览到 **/predict 时返回预测值。**

int_features:这里我们从 HTML 表单中取出所有的特性，并将其转换成 float。

final_features:这里我们将它们转换成 NumPy 数组。这样我们就可以直接预测值了。

预测:在加载模型的帮助下，我们正在进行预测

输出:这里我们将输出保存为一个字符串

return:这里我们将输出返回给 index.html

__name__ == "__main__ ":在这里，我们终于在这个里面运行应用程序了。

# 网站(全球资讯网的主机站)

![](img/f89eef2467dbae66a2d513cd47c4038e.png)

由[格伦·卡斯滕斯-彼得斯](https://unsplash.com/@glenncarstenspeters?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

现在让我们快速看看我们将如何创建我们的网站。

现在创建一个**模板文件夹**，在里面你需要创建**index.html**

书写我们的 index.html

```
<!DOCTYPE html><html ><head><meta charset="UTF-8"><title>ML API</title></head><body><div class="login"><h1>Predict Salary Analysis</h1><!-- Main Input For Receiving Query to our ML --><form action="{{ url_for('predict')}}"method="post"><input type="text" name="SepalLength" placeholder="SepalLength" required="required" /><input type="text" name="SepalWidth" placeholder="SepalWidth" required="required" /><input type="text" name="PetalLength" placeholder="PetalLength" required="required" /><input type="text" name="PetalWidth" placeholder="PetalWidth" required="required" /><button type="submit" class="btn btn-primary btn-block btn-large">Predict</button></form><br><br>{{ prediction_text }}</div></body></html>
```

*在这个简单的 HTML 中，我们创建了一个包含所有文本类型输入的简单表单，当我们填充所有表单时，当我们单击 predict***{ { URL _ for(' predict ')} }**，这是一个 **Jinja** ，意思是 **/predict。**

一旦用户提交表单，app.py 中的预测函数就会被撤销，然后它会返回预测值，该值显示在我们的 index.html 中，使用的是{{ prediction_text }}。

# 结论

这就是全部。如果你想要完整的代码，它在我的 Github 仓库里，我也在 Heroku 上部署了它。你也可以添加更多的 HTML 和 CSS，使网站更具互动性。我会写更多关于你如何做任何 NLP 项目的相同过程的帖子，所以一旦我这样做了，请确保你跟随我获得更新。

[](https://github.com/CreatorGhost/IriisApi) [## CreatorGhost/IriisApi

### 这是一个简单的网络应用程序，由机器学习驱动，从最受欢迎的虹膜数据集预测花的类型…

github.com](https://github.com/CreatorGhost/IriisApi)