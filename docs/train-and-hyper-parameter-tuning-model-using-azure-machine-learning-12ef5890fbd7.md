# 使用 Azure 机器学习训练和超参数调整模型

> 原文：<https://medium.com/analytics-vidhya/train-and-hyper-parameter-tuning-model-using-azure-machine-learning-12ef5890fbd7?source=collection_archive---------26----------------------->

![](img/27b473fc8703afcca83662c630091e8e.png)

# 用例

使用 scikit RandomClassifier 训练基本模型，验证并获得准确性，然后运行超参数调整。
在这个例子中，我们将使用 Azure 机器学习管道来训练我们可以用于 CI/CD 的 Azure DevOps

# 步伐

首先训练一个基本模型。

让我们导入所有必需的包

```
import osimport urllibimport shutilimport azuremlimport pandas as pdfrom azureml.core import Experimentfrom azureml.core import Workspace, Runfrom azureml.core.compute import ComputeTarget, AmlComputefrom azureml.core.compute_target import ComputeTargetExceptionfrom azureml.core import Experiment, Workspace, Run, Datasetimport argparseimport os
import urllib
import shutil
import azureml
import pandas as pdfrom azureml.core import Experiment
from azureml.core import Workspace, Runfrom azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment, Workspace, Run, Datasetimport argparse
import os
import pandas as pd
import numpy as np
import pickle
import jsonimport sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModelimport azureml.core
from azureml.core import Run
from azureml.core.model import Model
from azureml.core import Workspace, Dataset
from azureml.core import Experiment
from azureml.core import Workspace, Runfrom azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
```

设置身份验证以使用服务主体运行模型

```
import os
from azureml.core.authentication import ServicePrincipalAuthentication

svc_pr_password = os.environ.get("AZUREML_PASSWORD")

svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxx")

ws = Workspace(
    subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    resource_group="mlops",
    workspace_name="mlopsdev",
    auth=svc_pr
    )

print("Found workspace {} at location {}".format(ws.name, ws.location))
```

现在是时候设置要使用的环境和项目文件夹了

```
project_folder = './diabetes-project'
os.makedirs(project_folder, exist_ok=True)experiment = Experiment(workspace=ws, name='diabetes-model')output_folder = './outputs'
os.makedirs(output_folder, exist_ok=True)result_folder = './results'
os.makedirs(result_folder, exist_ok=True)
```

现在加载数据:(这个数据设置是公开的，没有真实的数据)

```
df = pd.read_csv('https://mlopssa.blob.core.windows.net/chd-dataset/framingham.csv')
```

现在是时候清理数据集，只获取我们需要的特性了

```
# create a boolean array of smokers
smoke = (df['currentSmoker']==1)
# Apply mean to NaNs in cigsPerDay but using a set of smokers only
df.loc[smoke,'cigsPerDay'] = df.loc[smoke,'cigsPerDay'].fillna(df.loc[smoke,'cigsPerDay'].mean())# Fill out missing values
df['BPMeds'].fillna(0, inplace = True)
df['glucose'].fillna(df.glucose.mean(), inplace = True)
df['totChol'].fillna(df.totChol.mean(), inplace = True)
df['education'].fillna(1, inplace = True)
df['BMI'].fillna(df.BMI.mean(), inplace = True)
df['heartRate'].fillna(df.heartRate.mean(), inplace = True)# Features and label
features = df.iloc[:,:-1]
result = df.iloc[:,-1] # the last column is what we are about to forecast
```

分割数据集

```
# Train & Test splitX_train, X_test, y_train, y_test = train_test_split(features, result, test_size = 0.2, random_state = 14)
```

火车模型

```
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)clf.fit(X_train, y_train)
```

运行特征重要性并过滤掉不需要的列

```
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.12
sfm = SelectFromModel(clf, threshold=0.12)# Train the selector
sfm.fit(X_train, y_train)# Features selected
featureNames = list(features.columns.values) # creating a list with features' names
print("Feature names:")
for featureNameListindex in sfm.get_support(indices=True):
    print(featureNames[featureNameListindex])# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]# With only imporant features. Can check X_important_train.shape[1]
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)
#Y_important_test = sfm.transform(y_test)rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
rfc.fit(X_important_train, y_train)
```

现在用剩余数据预测

```
preds = rfc.predict(X_important_test)
```

从上述模型中获取指标

```
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, preds))
```

现在是时候设置管道了。让我们来配置计算

```
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException# Choose a name for your CPU cluster
cpu_cluster_name = "diabetescluster"# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D14_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)cpu_cluster.wait_for_completion(show_output=True)
```

设置计算变量和环境

```
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE# Create a new runconfig object
run_amlcompute = RunConfiguration()# Use the cpu_cluster you created above. 
run_amlcompute.target = cpu_cluster# Enable Docker
run_amlcompute.environment.docker.enabled = True# Set Docker base image to the default CPU-based image
run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE# Use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_amlcompute.environment.python.user_managed_dependencies = False# Specify CondaDependencies obj, add necessary packages
run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])
```

写入 train.py 文件的时间

```
%%writefile $project_folder/train.pyimport joblib
import os
import urllib
import shutil
import azureml
import argparse
import pandas as pd
import numpy as np
import pickle
import json
from sklearn import metricsimport sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModelfrom azureml.core import Experiment
from azureml.core import Workspace, Runfrom azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetExceptionfrom sklearn.ensemble import RandomForestClassifierfrom azureml.core import Workspace, Datasetfrom azureml.core.authentication import ServicePrincipalAuthentication

svc_pr_password = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

svc_pr = ServicePrincipalAuthentication(
    tenant_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    service_principal_password="xxxxxxxxxxxxxxxxxxxxxxxx")

ws = Workspace(
    subscription_id="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    resource_group="mlops",
    workspace_name="mlopsdev",
    auth=svc_pr
    )#dataset = Dataset.get_by_name(ws, name='touringdataset')
#dataset.to_pandas_dataframe()
data_complete = df = pd.read_csv('[https://mlopssa.blob.core.windows.net/chd-dataset/framingham.csv'](https://mlopssa.blob.core.windows.net/chd-dataset/framingham.csv'))# Get the experiment run context
run = Run.get_context()# create a boolean array of smokers
smoke = (df['currentSmoker']==1)
# Apply mean to NaNs in cigsPerDay but using a set of smokers only
df.loc[smoke,'cigsPerDay'] = df.loc[smoke,'cigsPerDay'].fillna(df.loc[smoke,'cigsPerDay'].mean())# Fill out missing values
df['BPMeds'].fillna(0, inplace = True)
df['glucose'].fillna(df.glucose.mean(), inplace = True)
df['totChol'].fillna(df.totChol.mean(), inplace = True)
df['education'].fillna(1, inplace = True)
df['BMI'].fillna(df.BMI.mean(), inplace = True)
df['heartRate'].fillna(df.heartRate.mean(), inplace = True)# Features and label
features = df.iloc[:,:-1]
result = df.iloc[:,-1] # the last column is what we are about to forecast# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size = 0.2, random_state = 14)# RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.12
sfm = SelectFromModel(clf, threshold=0.12)# Train the selector
sfm.fit(X_train, y_train)# Features selected
featureNames = list(features.columns.values) # creating a list with features' names
print("Feature names:")
for featureNameListindex in sfm.get_support(indices=True):
    print(featureNames[featureNameListindex])# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]# With only imporant features. Can check X_important_train.shape[1]
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
rfc.fit(X_important_train, y_train)preds = rfc.predict(X_important_test)run.log("Accuracy:",metrics.accuracy_score(y_test, preds))print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))#joblib.dump(rfc, "/outputs/model.joblib")
os.makedirs('./outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=rfc, filename='./outputs/sklearn_diabetes_model.pkl')
```

创建估算器来运行管道

```
from azureml.train.sklearn import SKLearnestimator = SKLearn(source_directory=project_folder, 
#                     script_params=script_params,
                    compute_target=cpu_cluster,
                    entry_script='train.py',
                    pip_packages=['joblib']
                   )
```

提交并运行实验。

```
run = experiment.submit(estimator) run.wait_for_completion(show_output=True)
```

打印来自模型的指标

```
print(run.get_metrics())
```

应该是类似于{ ' Accuracy:':0.82339622641 }

显示结果和进度

```
from azureml.widgets import RunDetails
RunDetails(run).show()
```

现在是时候超调参数来验证模型了

```
from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, loguniformparam_sampling = GridParameterSampling( {
        "num_hidden_layers": choice(1, 2, 3),
        "batch_size": choice(16, 32)
    }
)
```

设定参数

```
primary_metric_name="accuracy",
primary_metric_goal=PrimaryMetricGoal.MAXIMIZE
```

设置强盗政策

```
from azureml.train.hyperdrive import BanditPolicy
early_termination_policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)
```

设置截断策略

```
from azureml.train.hyperdrive import TruncationSelectionPolicy
early_termination_policy = TruncationSelectionPolicy(evaluation_interval=1, truncation_percentage=20, delay_evaluation=5)
```

设置超调参数的退出标准

```
max_total_runs=20,
max_concurrent_runs=4
```

设置超参数调整运行

```
from azureml.train.hyperdrive import HyperDriveConfig
hyperdrive_run_config = HyperDriveConfig(estimator=estimator,
                          hyperparameter_sampling=param_sampling, 
                          policy=early_termination_policy,
                          primary_metric_name="accuracy",
                     primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                          max_total_runs=100,
                          max_concurrent_runs=4)
```

提交并运行超级参数调整

```
from azureml.core.experiment import Experiment
experiment = Experiment(workspace=ws, name='diabetes-model')
hyperdrive_run = experiment.submit(hyperdrive_run_config)
```

显示状态

```
from azureml.widgets import RunDetails
RunDetails(hyperdrive_run).show()
```

获得性能最佳的模型

```
best_run = hyperdrive_run.get_best_run_by_primary_metric()
sprint(best_run.get_details()['runDefinition']['arguments'])
```

显示模型

```
print(best_run.get_file_names())
```

现在是注册模特的时候了

```
model = best_run.register_model(model_name='sklearn_diabetes',
                           model_path='outputs/sklearn_diabetes_model.pkl',
                           tags=run.get_metrics())
print(model.name, model.id, model.version, sep='\t')
```

最佳模特将被注册。现在，该模型可以用于部署。

*原载于【https://github.com】[](https://github.com/balakreshnan/mlops/blob/master/traininghypertuning.md)**。***