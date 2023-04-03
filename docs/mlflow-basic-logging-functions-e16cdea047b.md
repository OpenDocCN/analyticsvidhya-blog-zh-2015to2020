# MLflow:基本日志功能

> 原文：<https://medium.com/analytics-vidhya/mlflow-basic-logging-functions-e16cdea047b?source=collection_archive---------8----------------------->

![](img/b3b4593ca9a7ec19626adced49c04532.png)

这是我的 MLflow 教程系列的第二篇文章:

1.  [在生产中设置 ml flow](/@gyani91/setup-mlflow-in-production-d72aecde7fef)
2.  [MLflow:基本的日志功能](/@gyani91/mlflow-basic-logging-functions-e16cdea047b)(你来了！)
3.  [张量流的 MLflow 测井](/@gyani91/mlflow-logging-for-tensorflow-37b6a6a53e3c)
4.  [MLflow 项目](/@gyani91/mlflow-projects-24c41b00854)
5.  [使用 Python API 为 MLflow 检索最佳模型](/@gyani91/retrieving-the-best-model-using-python-api-for-mlflow-7f76bf503692)
6.  [使用 MLflow 服务模型](/@gyani91/serving-a-model-using-mlflow-8ba5db0a26c0)

让我们从一些基本的 MLflow 函数开始，这些函数将帮助您记录各种值和工件。

```
import mlflow
```

日志记录功能需要与特定的运行相关联。将所有内容放入一次运行的最佳方式是在 main 函数(或其他调用函数)的开始处指定运行的开始，在 main 函数的结束处指定运行的结束。

```
if __name__ == '__main__':
  mlflow.start_run()
  #the model code
  mlflow.end_run()
```

`[mlflow.log_param()](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param)`记录当前运行中的单个键值参数。键和值都是字符串。

```
mlflow.log_param('training_steps', training_steps)
```

`[mlflow.log_metric()](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric)`记录单个键值度量。该值必须始终是一个数字。

```
mlflow.log_metric('accuracy', accuracy)
```

`[mlflow.log_artifacts()](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifacts)`将一个给定目录中的所有文件记录为工件，可选的`artifact_path`。

```
mlflow.log_artifacts(export_path, "model")
```

上述语句将把 *export_path* 上的所有文件记录到 MLflow 运行的工件目录内名为 *"model"* 的目录中。

更多信息请参考[记录功能](https://www.mlflow.org/docs/latest/tracking.html#logging-functions)。

在下一篇文章的[中，我们将加深对 MLflow 测井函数的理解，并将它们用于 TensorFlow。](/@gyani91/mlflow-logging-for-tensorflow-37b6a6a53e3c)