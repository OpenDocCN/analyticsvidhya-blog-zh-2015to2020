# 使用 Azure ML 服务的对象检测— AutoML

> 原文：<https://medium.com/analytics-vidhya/object-detection-using-azure-ml-service-automl-b061035aa7d3?source=collection_archive---------19----------------------->

![](img/a9a56287bdeed708e80d7130b2a57038.png)

# 使用 Azure ML Assist 数据标签工具的输出并构建模型

了解如何轻松构建端到端对象检测，包括标记和建模。也正在部署。

# 先决条件

*   创建 blob 容器
*   创建列车文件夹
*   基于目录从[https://github . com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/AML tojson . MD](https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/AMLtoJson.md)移动所有图像
*   目录结构应遵循“AML datastore://storimagesjson/train/no stock/imagname . jpg”
*   对于上面的例子，storeimagesjson 是容器名
*   火车是主文件夹，我创建的类是另一个名为 nostock 的文件夹
*   然后把图像放在那里
*   对于注释文件，将其放置在 train 文件夹下
*   Azure ML 服务仅支持以下地区:eastus，eastus2
*   我只测试 eastus2

```
blobContainer 
 /train /nostock 
 /test /nostock 
 /annotation.jsonl
```

# 更新 azure ml sdk

```
pip install --upgrade azureml-sdk 
pip install --upgrade azureml-contrib-automl-dnn-vision 
print("SDK version:", azureml.core.VERSION)
```

# 构建新的模型训练代码。

```
import logging
import os
import csv

import numpy as np
import pandas as pd
from sklearn import datasets

import azureml.core
from azureml.core import Run, Workspace
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig
import azureml.dataprep as dprep
from azureml.core.dataset import Datasetws = Workspace.from_config()experiment_name = 'labeling_Training_xxxxxxx'
project_folder = './project'

experiment = Experiment(ws, experiment_name)

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace Name'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
output['Experiment Name'] = experiment.name
pd.set_option('display.max_colwidth', -1)
outputDf = pd.DataFrame(data = output, index = [''])
outputDf.T
```

*   实验名称是我从数据标签项目中提取的，在项目主页中，您应该可以看到列车运行实验名称。复制并粘贴到这里
*   设置计算

```
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

# Choose a name for your cluster.
amlcompute_cluster_name = "gpucluster"

found = False
# Check if this compute target already exists in the workspace.
cts = ws.compute_targets
if amlcompute_cluster_name in cts and cts[amlcompute_cluster_name].type == 'AmlCompute':
    found = True
    print('Found existing compute target.')
    compute_target = cts[amlcompute_cluster_name]

if not found:
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = "STANDARD_NC6",
                                                                max_nodes = 4)

    # Create the cluster.\n",
    compute_target = ComputeTarget.create(ws, amlcompute_cluster_name, provisioning_config)

    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min_node_count is provided, it will use the scale settings for the cluster.
    compute_target.wait_for_completion(show_output = True, min_node_count = None, timeout_in_minutes = 20)

     # For a more detailed view of current AmlCompute status, use get_status().from azureml.core.datastore import Datastore

account_key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ds = Datastore.register_azure_blob_container(ws, datastore_name='containername', container_name='containername', 
                                             account_name='storageaccountname', account_key=account_key,
                                             resource_group='resourcegroupname')from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
import pkg_resources

# create a new RunConfig object
conda_run_config = RunConfiguration(framework="python")

# Set compute target to AmlCompute
conda_run_config.target = compute_target
conda_run_config.environment.docker.enabled = True
conda_run_config.environment.docker.base_image = azureml.core.runconfig.DEFAULT_GPU_IMAGE

cd = CondaDependencies()

conda_run_config.environment.python.conda_dependencies = cdfrom azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory
from azureml.core import Dataset

# create training dataset
training_dataset_name = experiment_name + "_training_dataset"
if training_dataset_name in ws.datasets:
    training_dataset = ws.datasets.get(training_dataset_name)
    print('Found the dataset', training_dataset_name)
else:
    training_dataset = _LabeledDatasetFactory.from_json_lines(
        task="ImageClassification", path=ds.path('train/annotation.jsonl'))
    training_dataset = training_dataset.register(workspace=ws, name=training_dataset_name)automl_settings = {
    "iteration_timeout_minutes": 1000,
    "iterations": 1,
    "primary_metric": 'mean_average_precision',
    "featurization": 'off',
    "enable_dnn": True,
    "dataset_id": training_dataset.id
}
automl_config = AutoMLConfig(task = 'image-object-detection',
                             debug_log = 'automl_errors_1.log',
                             path = project_folder,
                             run_configuration=conda_run_config,
                             training_data = training_dataset,
                             label_column_name = "label",
                             **automl_settings
                            )remote_run = experiment.submit(automl_config, show_output = False)remote_run
remote_run.wait_for_completion()Part 3: Deploying the above model to REST API[https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/MLAssitObjScoring.md](https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/MLAssitObjScoring.md)Share your thoughts and comments.
```

*最初发表于*[T5【https://github.com】](https://github.com/balakreshnan/mlops/blob/master/MLAssitDataLabelling/MLAssitObjDetectionTraining.md)*。*