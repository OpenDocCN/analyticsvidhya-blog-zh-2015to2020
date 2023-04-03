# MLflow 项目

> 原文：<https://medium.com/analytics-vidhya/mlflow-projects-24c41b00854?source=collection_archive---------6----------------------->

![](img/b3b4593ca9a7ec19626adced49c04532.png)

这是我的 MLflow 教程系列的第四篇文章:

1.  [在生产中设置 ml flow](/@gyani91/setup-mlflow-in-production-d72aecde7fef)
2.  [MLflow:基本测井功能](/@gyani91/mlflow-basic-logging-functions-e16cdea047b)
3.  [张量流的 MLflow 测井](/@gyani91/mlflow-logging-for-tensorflow-37b6a6a53e3c)
4.  [MLflow 项目](/@gyani91/mlflow-projects-24c41b00854)(你来了！)
5.  [使用 Python API 为 MLflow 检索最佳模型](/@gyani91/retrieving-the-best-model-using-python-api-for-mlflow-7f76bf503692)
6.  [使用 MLflow 服务模型](/@gyani91/serving-a-model-using-mlflow-8ba5db0a26c0)

如果你创建一个新的项目或者克隆一个现有的项目，你可以通过简单地添加两个 YAML 文件，即。、[ml 项目文件](https://www.mlflow.org/docs/latest/projects.html#mlproject-file)和 [Conda](https://conda.io/projects/conda/en/latest/) 环境文件，放在项目的根目录下。

这一步不是必须的，但是强烈推荐，因为它不仅增强了模型的可再现性，而且将运行链接到代码的特定版本(它的 git 散列)。这非常有用，因为如果将来对代码的更改影响了它的功能和/或结果，用户可以简单地对特定的 git 提交进行 git 检验。

Cityscapes 语义分段数据集上 [Deeplab 的 MLproject 文件示例:](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md)

```
name: deeplab
conda_env: conda.yamlentry_points:
  main:
    parameters:
      training_number_of_steps: {type: int, default: 900}
      output_stride: {type: int, default: 16}
      decoder_output_stride: {type: int, default: 4}
      train_batch_size: {type: int, default: 1}
      dataset: {default: 'cityscapes'}
      train_logdir: {default: /home/sumeet/models/research/deeplab/datasets/cityscapes/exp/train_on_train_set/train}
      dataset_dir: {default: /home/sumeet/models/research/deeplab/datasets/cityscapes/tfrecord}
    command: "python train.py \
    --logtostderr \
    --training_number_of_steps={training_number_of_steps} \
    --train_split='train' \
    --model_variant='xception_65' \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride={output_stride} \
    --decoder_output_stride={decoder_output_stride} \
    --train_crop_size='769,769' \
    --train_batch_size={train_batch_size} \
    --dataset={dataset} \
    --train_logdir={train_logdir} \
    --dataset_dir={dataset_dir}"
```

Conda 环境文件的示例:

```
name: production_env
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - ca-certificates=2019.8.28=0
  - certifi=2019.9.11=py37_0
  - libedit=3.1.20181209=hc058e9b_0
  - libffi=3.2.1=hd88cf55_4
  - libgcc-ng=9.1.0=hdf63c60_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - ncurses=6.1=he6710b0_1
  - openssl=1.1.1d=h7b6447c_2
  - pip=19.2.3=py37_0
  - python=3.7.4=h265db76_1
  - readline=7.0=h7b6447c_5
  - sqlite=3.30.0=h7b6447c_0
  - tk=8.6.8=hbc83047_0
  - xz=5.2.4=h14c3975_4
  - zlib=1.2.11=h7b6447c_3
  - pip:
    - absl-py==0.8.1
    - alembic==1.2.1
    - astor==0.8.0
    - attrs==19.2.0
    - backcall==0.1.0
    - bleach==3.1.0
    - chardet==3.0.4
    - cityscapesscripts==1.1.0
    - click==7.0
    - cloudpickle==1.2.2
    - configparser==4.0.2
    - cycler==0.10.0
    - databricks-cli==0.9.0
    - decorator==4.4.0
    - defusedxml==0.6.0
    - docker==4.1.0
    - entrypoints==0.3
    - flask==1.1.1
    - gast==0.2.2
    - gitdb2==2.0.6
    - gitpython==3.0.3
    - google-pasta==0.1.7
    - gorilla==0.3.0
    - grpcio==1.24.1
    - gunicorn==19.9.0
    - h5py==2.10.0
    - idna==2.8
    - imdbclassifier==0.6.6
    - importlib-metadata==0.23
    - ipykernel==5.1.2
    - ipython==7.8.0
    - ipython-genutils==0.2.0
    - ipywidgets==7.5.1
    - itsdangerous==1.1.0
    - jedi==0.15.1
    - jinja2==2.10.3
    - joblib==0.14.0
    - jsonschema==3.1.1
    - jupyter==1.0.0
    - jupyter-client==5.3.4
    - jupyter-console==6.0.0
    - jupyter-core==4.6.0
    - keras==2.3.1
    - keras-applications==1.0.8
    - keras-preprocessing==1.1.0
    - kiwisolver==1.1.0
    - mako==1.1.0
    - markdown==3.1.1
    - markupsafe==1.1.1
    - matplotlib==3.1.1
    - mistune==0.8.4
    - mlflow==1.3.0
    - more-itertools==7.2.0
    - nbconvert==5.6.0
    - nbformat==4.4.0
    - notebook==6.0.1
    - numpy==1.17.2
    - opt-einsum==3.1.0
    - pandas==0.25.1
    - pandocfilters==1.4.2
    - parso==0.5.1
    - pexpect==4.7.0
    - pickleshare==0.7.5
    - pillow==6.2.0
    - prettytable==0.7.2
    - prometheus-client==0.7.1
    - prompt-toolkit==2.0.10
    - protobuf==3.10.0
    - ptyprocess==0.6.0
    - pygments==2.4.2
    - pyparsing==2.4.2
    - pyrsistent==0.15.4
    - python-dateutil==2.8.0
    - python-editor==1.0.4
    - pytz==2019.3
    - pyyaml==5.1.2
    - pyzmq==18.1.0
    - qtconsole==4.5.5
    - querystring-parser==1.2.4
    - requests==2.22.0
    - scikit-learn==0.21.3
    - scipy==1.3.1
    - send2trash==1.5.0
    - setuptools==41.4.0
    - simplejson==3.16.0
    - six==1.12.0
    - sklearn==0.0
    - smmap2==2.0.5
    - sqlalchemy==1.3.9
    - sqlparse==0.3.0
    - tabulate==0.8.5
    - tensorboard==1.15.0
    - tensorflow==1.15.0
    - tensorflow-estimator==1.15.1
    - tensorflow-gpu==1.15.0
    - termcolor==1.1.0
    - terminado==0.8.2
    - testpath==0.4.2
    - tornado==6.0.3
    - traitlets==4.3.3
    - urllib3==1.25.6
    - wcwidth==0.1.7
    - webencodings==0.5.1
    - websocket-client==0.56.0
    - werkzeug==0.16.0
    - wheel==0.33.6
    - widgetsnbextension==3.5.1
    - wrapt==1.11.2
    - zipp==0.6.0
prefix: ~/anaconda3/envs/production_env
```

您可以手动创建上述文件，或者如果您已经有一个稳定的 Conda 环境，您可以使用以下命令将其导出到一个文件中:

```
conda env export > conda.yaml
```

更多信息，请参考[物流项目](https://www.mlflow.org/docs/latest/projects.html)。

在下一篇文章中，我将向您展示如何使用 Python API 从 MLflow 的实验中检索最佳模型。