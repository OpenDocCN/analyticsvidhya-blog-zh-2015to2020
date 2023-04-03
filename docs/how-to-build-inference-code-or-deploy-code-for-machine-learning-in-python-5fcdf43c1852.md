# 在 python 中为机器学习创建推理代码和部署代码

> 原文：<https://medium.com/analytics-vidhya/how-to-build-inference-code-or-deploy-code-for-machine-learning-in-python-5fcdf43c1852?source=collection_archive---------0----------------------->

由于致力于机器学习和人工智能，我看到的挑战是如何部署机器学习运行时或评分模型以及作为 rest API 运行存在差距。有很多关于如何构建和训练模型的文档。还有一些方法可以运行与批处理相同的模型。但是问题是如何将模型作为 rest api 来操作。现在，随着 DevOps 实践和行业将每个人都推向 docker，问题变成了如何在 docker 容器中部署模型并运行。

现在 python 本身是开源的，存在 2 和 3 版本冲突的问题。Docker 也是开源的。鉴于所有这些工具都是开源的，不确定它会有多稳定和可靠。

这是我试图创造的。建立一个简单的机器学习分类模型。本教程不是关于构建模型，而是我们如何使用 pickle 保存训练好的模型，然后使用 Flask 编写推理代码使其成为 rest API。以下是细节

1.  将使用 pickle 构建的模型保存为 pickle 文件
2.  用 python 创建一个推理代码来读取 pickle 文件，并用新数据进行评分。
3.  构建 Docker 文件，从推理代码创建一个容器。
4.  构建包含所有依赖项的 requirements.txt 文件。
5.  构建 docker 容器。

1.将使用 pickle 构建的模型保存为 pickle 文件

我在用可视化代码写机器学习 python 代码。要保存模型，请确保您

导入 pickle #这一步应该在模型构建文件的开头。在调用 mode.fit 的那一行之后(在这种情况下，我使用 scikit-learn ),然后调用 pickle 保存模型，下面是示例代码:

#我们创建一个分类器实例并拟合数据。

clf.fit(X_scaled，y)

#将模型保存为 pickle 文件

filename = 'carlr.pkl '

pickle.dump(clf，open(文件名，' wb ')，协议=2)

添加这些代码并运行模型 python 文件后，pickle 文件应该会弹出。为模型选择您自己的文件名。我使用 protocol = 2，因为它兼容 python 2 和 3 版本。我在知道我需要包括什么包和协议版本方面有一些挑战，所以想分享我的知识。

2.用 python 创建一个推理代码来读取 pickle 文件，并用新数据进行评分。

现在是编写推理代码的时候了。下面是我写的样本代码。我在 azure 中使用数据科学虚拟机，因为所有工具都可用。我将以下内容保存到 mltestweb.py python 文件中。

#步骤 1 —加载数据

########################

进口泡菜

进口熊猫作为 pd

导入操作系统、uuid、系统

#从 azure.storage.blob 导入 BlockBlobService，PublicAccess

将 numpy 作为 np 导入

从烧瓶导入烧瓶，请求

#http://127.0.0.1:80/predict？价格= 13500 &年龄= 23 & KM = 46986 & fuel type = Diesel & HP = 90 & met color = 1 & Automatic = 0 & CC = 2000 & Doors = 4 & Weight = 1165

#13500，2346986，柴油，90，1，0，2000，31165

#创建要运行的 Flask 对象

app = Flask(__name__)

@app.route('/')

def home():

return“嗨，欢迎来到 Flask！!"

@app.route('/predict ')

定义预测():

#从浏览器获取值

#价格、车龄、公里数、燃料类型、马力、MetColor、自动、CC、车门、重量

#在 GET 请求中读取 URL 中的所有请求变量。

价格= request.args['价格']

Age = request.args['Age']

KM = request.args['KM']

fuel type = request . args[' fuel type ']

HP = request.args['HP']

met color = request . args[' met color ']

Automatic = request . args[' Automatic ']

CC = request.args['CC']

Doors = request.args['Doors']

Weight = request.args['Weight']

# test data = NP . array([价格，年龄，公里，燃料类型，HP，MetColor，自动，CC，车门，重量])。整形(1，10)

#将变量转换成数据帧。

testData = pd。DataFrame({'Age ':年龄，' KM': KM}，索引=[0])

X = testData[["Age "，" KM"]]。价值观念

y=np.array([0])

来自 sklearn .预处理导入标准缩放器

scaler = StandardScaler()

X _ scaled = scaler . fit _ transform(X)

#从磁盘加载模型

filename = 'carlr.pkl '

loaded_model = pickle.load(open(文件名，' rb '))

result = loaded _ model . score(X _ scaled，y)

output = "Age:" + str(年龄)+"，KM:" + str(KM) +"，实惠:"+ str(结果)

#output = "预测汽车价格: "

返回(输出)

# Main 方法。

if __name__ == '__main__ ':

#运行样本()

打印(" * *启动服务器…")

#运行服务器

app.run(主机='0.0.0.0 '，端口=80)

3.构建 Docker 文件，从推理代码创建一个容器。

我使用 linux 数据科学虚拟机来创建 docker 文件。需要注意的一点是确保 dockerfile 的文件名都是小写的，因为 DockerFile 对我不适用。

以下是 docker 文件的内容:

来自 ubuntu:最新

维护者 xxxx " xxx @ yyyyyyyyy.com "

运行 apt-get update -y

运行 apt-get install-y python-pip python-dev build-essential

暴露 80

收到。/应用程序

工作目录/应用程序

运行 pip install -r requirements.txt

入口点["python"]

CMD ["mltestweb.py"]

4.构建包含所有依赖项的 requirements.txt 文件。

requirements.txt 文件列出了 scikit 和 flask 所需的所有包。这是这个文件的内容

烧瓶==0.12.2

熊猫==0.23.4

scikit-learn==0.19.1

scipy==1.1.0

numpy==1.14.6

5.构建 docker 容器

现在复制 pickle 文件、推理代码 flask python 文件、dockerfile 和 requirements.txt，并将其移动到 linux vm。您可以使用 winscp 或 ftp 或 ssh 来复制文件。现在 ssh 到您复制文件的 linux 机器中。

现在，我们先来看看如何创建 docker 图像。我们需要指定 dockerfile 的位置，并用适当的名称标记图像。mlsample:latest 表示容器的名称以及当前版本。我在/data/home/mlsample/文件夹中有所有的模型文件和 docker 文件。首先转到文件夹，然后运行命令。

sudo docker build/data/home/ml sample/-t ml sample:最新

下一步是运行模型。

sudo docker run-d-p 80:80ml 样品

现在，如果您键入 sudo docker ps -a，您应该会看到容器正在运行。

下一步是将容器推到某个容器注册中心，然后从那里可以像使用 AKS 或任何其他容器部署工具和技术的任何其他微服务部署一样进行部署。

要推送注册，首先使用 sudo docker ps 找到容器 id

然后首先提交映像，使用:

```
sudo docker commit c16378f943fe mysample
```

现在这样做，以推动注册。

```
$ sudo docker tag rhel-httpd registry-host:5000/myadmin/mysample

$ sudo docker push registry-host:5000/myadmin/mysample
```

如果这是私人注册，你需要认证，以推动图像注册。

1.  `sudo docker login --username username --password password`
2.  `sudo docker tag mysample username/mysample`
3.  `sudo docker push username/mysample`

从这一点上来说，它变成了一个 REST API Docker 容器，可以像部署其他微服务一样进行部署。现在，要在生产中部署 REST API，请确保您有足够的能力来满足您的客户需求。

一旦您部署了 REST API，您也可以使用任何浏览器或 postman 来测试您的 API。根据你的主机位置，你可能需要使用其他工具来监控和限制 API 的使用。如果这是公司内部使用案例，则没有必要。此外，如果你想推理代码安全，你也可以包括获得认证，并建立自己的和其他加密和其他合规技术。

这份文件作为一个模板，并享受机器和深度学习的乐趣。