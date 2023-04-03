# 使用 MLflow 服务模型

> 原文：<https://medium.com/analytics-vidhya/serving-a-model-using-mlflow-8ba5db0a26c0?source=collection_archive---------4----------------------->

![](img/b3b4593ca9a7ec19626adced49c04532.png)

这是我的 MLflow 教程系列的第六篇也是最后一篇文章:

1.  [在生产中设置 ml flow](/@gyani91/setup-mlflow-in-production-d72aecde7fef)
2.  [MLflow:基本测井功能](/@gyani91/mlflow-basic-logging-functions-e16cdea047b)
3.  [张量流的 MLflow 测井](/@gyani91/mlflow-logging-for-tensorflow-37b6a6a53e3c)
4.  [MLflow 项目](/@gyani91/mlflow-projects-24c41b00854)
5.  [使用 Python API 为 MLflow 检索最佳模型](/@gyani91/retrieving-the-best-model-using-python-api-for-mlflow-7f76bf503692)
6.  [使用 MLflow 为模型服务](/@gyani91/serving-a-model-using-mlflow-8ba5db0a26c0)(你来了！)

创造环境

```
conda create -n production_env
conda activate production_env
conda install python
pip install mlflow
pip install sklearn
```

从互联网上运行一个样本机器学习模型

```
mlflow run git@github.com:databricks/mlflow-example.git -P alpha=0.5
```

注:正如 [Mourad K](https://medium.com/u/6708ecbce7e4?source=post_page-----8ba5db0a26c0--------------------------------) 在评论中指出的。只有当您的 [GitHub 认证](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)使用 ssh-keys 设置时，上面的命令才会运行。

检查它是否成功运行

```
ls -al ~/mlruns/0
```

从上面的命令中获取我们刚刚运行的模型的 **uuid** 并为模型提供服务。

```
mlflow models serve -m ~/mlruns/0/your_uuid/artifacts/model -h 0.0.0.0 -p 8001
```

在新的终端窗口中进行推理。去狂野吧！

```
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' [http://0.0.0.0:8001/invocations](http://0.0.0.0:8001/invocations)
```

要使用 Python 进行推理，您可以导入**请求**库:

```
import requests

host = '0.0.0.0'
port = '8001'

url = f'[http://{host}:{port}/invocations](/analytics-vidhya/{host}:{port}/invocations)'

headers = {
    'Content-Type': 'application/json',
}

# test_data is a Pandas dataframe with data for testing the ML model
http_data = test_data.to_json(orient='split')

r = requests.post(url=url, headers=headers, data=http_data)

print(f'Predictions: {r.text}')
```

当您按下 Ctrl+C 或退出终端时， *mlflow 模型服务于*命令。如果您希望模型启动并运行，您需要为它创建一个 **systemd** 服务。进入 **/etc/systemd/system** 目录，新建一个名为 **model.service** 的文件，内容如下:

```
[Unit]
Description=MLFlow Model Serving
After=network.target

[Service]
Restart=on-failure
RestartSec=30
StandardOutput=file:/path_to_your_logging_folder/stdout.log
StandardError=file:/path_to_your_logging_folder/stderr.log
Environment=MLFLOW_TRACKING_URI=[http://host_ts:port_ts](http://host_ts:port_ts)
Environment=MLFLOW_CONDA_HOME=/path_to_your_conda_installation
ExecStart=/bin/bash -c 'PATH=/path_to_your_conda_installation/envs/model_env/bin/:$PATH exec mlflow models serve -m path_to_your_model -h host -p port'

[Install]
WantedBy=multi-user.target
```

使用以下命令激活并启用上述服务:

```
sudo systemctl daemon-reload
sudo systemctl enable model
sudo systemctl start model
sudo systemctl status model
```

上面的例子非常简单。对于像 [Deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) 这样的复杂模型，需要在模型保存期间定义输入和输出张量。为此，考虑参考 [TF 发球](https://www.tensorflow.org/tfx/guide/serving)。[这篇博客](https://www.freecodecamp.org/news/how-to-deploy-tensorflow-models-to-production-using-tf-serving-4b4b78d41700/)是使用 TF 上菜的 [Deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) 模型上菜的好指南。

# 参考资料:

[https://the gurus . tech/posts/2019/06/ml flow-production-setup/](https://thegurus.tech/posts/2019/06/mlflow-production-setup/)