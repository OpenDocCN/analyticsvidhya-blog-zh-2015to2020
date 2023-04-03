# 使用 Azure 机器学习工作区的 5 个简单步骤部署您的 ML 模型

> 原文：<https://medium.com/analytics-vidhya/deploy-your-ml-models-using-5-easy-steps-with-azure-machine-learning-workspace-c1ca5b6aa284?source=collection_archive---------12----------------------->

![](img/82dd5ac048e934be4b463180155ba194.png)

如果你正在读这篇文章，你要么已经开始了旅程，要么已经绝望了。好了，现在你有了最棒的机器学习模型，现在你可以预测股票市场了。你就要成为百万富翁了。但为了实现这一点，还有最后一个障碍需要跨越，那就是部署你的模型，也就是你的印钞机(简称 MP)。

在这一点上，你还决定使用云平台，因为你想让它不断运行，因为你知道…钱不会睡觉。所以，明智的选择。让我们把这个云叫做微软 Azure。你做了一点阅读，找到了你想要的东西，Azure Machine Learning Workspace(简称 AMLW 读作 Am-low)。好吧，我们离你的百万富翁越来越近了，我们真的很近了。有多近？大约五步之遥。所以让我们得到这笔钱。

# 步骤 0:包含您的库

作为程序员，我们总是从 0 开始。

```
#included libraries
import azureml.core
from azureml.core import Workspace, Dataset, Model, Webservice, Environment
from azureml.core.conda_dependencies import CondaDependencies
import json
import pandas as pd#check the azureml core being used
azureml.core.VERSION
```

# 步骤 1:初始化工作区

当您使用 Workspace.from_config()命令时，它会连接到笔记本所在的工作区。如果笔记本不在同一工作区，请手动指定工作区。我在这里包含了这两个选项。

选项 1:从配置

```
#inititalise workspace from config
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep=’\n’)
```

选项 2:手动初始化

```
#initialise workspace manually
subscription_id = ‘<enter user subscription Id>’
resource_group = ‘<name of resource group>’
workspace_name = ‘<name of azure machine learning resource name>’ws = Workspace(subscription_id, resource_group, workspace_name)
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep=’\n’)
```

# 步骤 2:连接到注册的模型

这一步假设模型已经被登记到模型存储中。如果这个假设是错误的，我需要把事情保持在 5 个步骤，所以你可以在这里找到细节

```
#Connect to model
modelname = ‘<name of model>’
model = Model(ws,modelname)
print(‘model name: ‘, model)
```

# 步骤 3:初始化环境

这是模型将要运行的容器。所以这可以是 Kubernetes 或者 Azure 容器实例(ACI)。对于运行在少量数据上的模型，我推荐使用 ACI。否则，使用 Kubernetes 选项。同样，在这里可以找到关于不同环境[的更多细节。](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where#deploy-to-target)

## 这一步声明了您的模型可能需要的所有依赖项。可以认为这是为您的模型安装的 pip

```
#Initialise the environment (pip packages to be installed, example shown below)
environment = Environment(‘<User chosen name>’)
environment.python.conda_dependencies = CondaDependencies.create(pip_packages=[
 ‘azureml-defaults’,
 ‘inference-schema[numpy-support]’,
 ‘joblib’,
 ‘scikit-learn’,
 ‘numpy==1.16.4’,
 ‘tensorflow==1.14’
])
```

请伙计们，非常重要的是，确保库版本与用于训练模型的库版本**完全**相同。如果没有，你可能会得到一个 502 错误，这可能是误导性的，并可能花费你生命中的几个小时…我想你可以告诉，我已经经历了火灾。

# 步骤 4:创建/导入 score.py 文件

分数文件基本上是在容器中运行模型的文件。因此，当数据被发送到部署的模型时，任何预处理或后处理步骤都可以在这个文件中完成。

我使用 Spyder 创建了 score.py 文件，但是，您可以使用您喜欢的文件，但是它必须是. py 格式

```
#create a score file variable as this is referenced in the deployment in step 5
score_file = ‘<name of score file>’
```

你可能在想，但是这个文件在哪里？事不宜迟…

```
import joblib
import os
import json#model specific libraries
import numpy as np 
import pandas as pddef init():
 ‘’’This function initialises the model. The model file is retrieved used
 within the script.
 ‘’’
 global model 

 model_filename = ‘<name of model file (.sav or .pkl)>’
 model_path = os.path.join(os.environ[‘AZUREML_MODEL_DIR’], model_filename)

 model = joblib.load(model_path)

#===================================================================
#Function definitions
#===================================================================

# Convert Series Data into Supervised Data
def functions(parameter):
 ‘’’
 Define functions neeeded in your model. If not needed, remove this part
 ‘’’#Run functiondef run(data):
 ‘’’
 Input the data as json and returns the predictions in json. All preprocessing 
 steps are specific to this model and usecase
 ‘’’
 #input data
 raw = pd.read_json(data) 

 #preprocessing steps 

 #prediction steps 
 pred = model.predict(data)

 #postprocessing steps #packaging steps

 result = pred.to_json()

 # You can return any JSON-serializable object.
 return result
```

使用下面的代码检查分数文件，以确保使用了正确的代码。我想安全总比后悔好。

```
#inspect the score script to ensure the correct script is being referenced
with open(score_file) as f:
    print(f.read())
```

# 步骤 5:将模型部署到环境中(将所有的东西放在一起)

每次运行这个步骤都需要一些时间。所以在你等待的时候，花点时间煮点咖啡吧。不过，加快速度的一个方法是在本地机器上部署 docker 容器，然后在将其推送到 ACI 之前在那里进行调试。

```
#deploy model to ACI
from azureml.core import Webservice
from azureml.core.model import InferenceConfig
from azureml.exceptions import WebserviceException#name given here will be the name of the deployed service
service_name = ‘<name of the deployed service>’# Remove any existing service under the same name.
try:
 Webservice(ws, service_name).delete()
except WebserviceException:
 passinference_config = InferenceConfig(entry_script= score_file,
 source_directory=’.’,
 environment=environment)service = Model.deploy(ws, service_name, [model], inference_config)
service.wait_for_deployment(show_output=True)
```

就是这样！您的容器应该部署到您的资源组中，并且应该在 AMLW 模型存储中创建一个 REST 端点。

但除此之外…钱！

# 不是步骤 0:测试

现在，我知道我说了 5 个步骤，但这只是测试，以确保一切正常。

为此，我们将连接到 AMLW 中的一个数据存储。有趣的事实是，这些数据集既可以手动加载到工作区，也可以引用数据存储，并在源位置连接数据，我认为这非常酷！aaaannnyyywaayy…我们继续

```
#connect to dataset
dataset = Dataset.get_by_name(ws, name=’<Name of dataset in AMLW>’)
dataset = dataset.to_pandas_dataframe()
```

使用您部署的模型，输入数据需要是 JSON 格式，然后输入到您的模型中。您可以使用下面的代码来打包并运行您的模型。

```
#package and run input data to model#input data
input_data = dataset.to_json()#run model
pred = service.run(input_data)#Convert returned json back to a pandas dataframe
pred = pd.read_json(pred)
```

# 不是步骤 1:使用端点

一旦部署了模型，就会创建一个 REST 端点，并且可以在您的其他平台中使用。下面的代码可以用在 Databricks 或其他 Python 环境中。

```
import requests
import json# URL for the web service
scoring_uri = ‘<Your deployment enpoint>’
# If the service is authenticated, set the key or token
key = ‘<your key or token>’# Convert to JSON string
input_data = dataset.to_json()# Set the content type
headers = {‘Content-Type’: ‘application/json’}
# If authentication is enabled, set the authorization header
headers[‘Authorization’] = f’Bearer {key}’# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.text)#load the returned prediction and read it into a pandas dataframe
pred = json.loads(resp.text)
pred = pd.read_json(pred)
```

故事就是这样。我认为这有点长，但公平地说，它的大部分是代码，所以如果我们仔细想想，它真的不算数。我应该提到，在 AMLW GitHub 和 docs 页面上可以找到很多这样的内容。如果你确实想学习更专业的东西，我建议你深入学习。那里有一些令人惊奇的东西，例如这里的。

我感谢你的阅读！享受你新发现的财富。