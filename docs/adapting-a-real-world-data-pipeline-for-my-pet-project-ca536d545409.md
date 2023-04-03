# 为我的宠物项目调整真实世界的数据管道

> 原文：<https://medium.com/analytics-vidhya/adapting-a-real-world-data-pipeline-for-my-pet-project-ca536d545409?source=collection_archive---------6----------------------->

![](img/045f6735dd28148885250df9a45e09e1.png)

昆腾·德格拉夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

Gitlab 有一个很酷的内部文化，默认情况下会把事情公开。这包括他们的整个[数据分析渠道](https://gitlab.com/gitlab-data/analytics)。我想熟悉他们的基础设施，所以我决定采用他们的工作，并根据我自己的目的进行调整。

前阵子，我写了一些代码从 Fitbit、Trace(滑雪板跟踪应用程序)和一些降雪数据中提取数据，以便对去年的滑雪季节进行一些数据分析。我决定自动接收 Fitbit 和 Trace 数据，并使用 DBT 而不是熊猫来进行转换。

(简化的)管道包括:

*   [Gitlab](https://gitlab.com/gitlab-data) :所有代码、CI 管道和图像库都托管在 Gitlab 中。
*   [谷歌 Kubernetes 引擎](https://cloud.google.com/kubernetes-engine/):气流在 GKE 上运行，使用 KubernetesPodOperators 完成大部分任务。
*   [雪花](https://www.snowflake.com/):数据仓库。存储从各种系统中提取的所有原始源数据以及由 DBT 创建的最终转换。
*   [气流](https://airflow.apache.org/):管理与提取、编排、数据库管理和转换(通过 DBT)相关的作业。
*   [DBT](https://www.getdbt.com/) :处理数据转换、数据谱系的文档，以及测试来自源和转换的数据质量。

# 设置回购

我创建了一个名为`snowboard-analysis`的 Gitlab 组，然后我将[分析](https://gitlab.com/snowboard-analysis/analytics)和[数据图像](https://gitlab.com/snowboard-analysis/data-image)回购分入其中。然后，我在`us-west1-a`中创建了一个名为`data-ops`的 Kubernetes 集群(参见“设置 GKE”)，其中有来自 [Gitlab group Kubernetes 页面](https://gitlab.com/groups/snowboard-analysis/-/clusters)的 6 个节点。我对名称和区域使用这些值，因为它们在一些地方是硬编码的，我不想改变它们。虽然 5 个节点就足够了，但是 6 个节点可以在同时运行多个任务时进行扩展。您也可以在 GKE 启用自动缩放来适应这一点。我还为 CI 创建了一个 [group runner，以便能够在我向 repo 推送更改时运行 CI/CD 脚本。](https://gitlab.com/groups/snowboard-analysis/-/settings/ci_cd)

在克隆了 repos 之后，我必须使用以下命令来更改硬编码的 repo 和 docker 注册表路径:

```
find . -type f -exec sed -i "s/gitlab.com\/gitlab-data/gitlab.com\/snowboard-analysis/g" {} \;
```

我还删除了所有不必要的 Dag、自定义提取、DBT 模型和测试、雪花角色、用户和数据库等。我留下了一些看起来有用的东西，比如一些与 DBT 和雪花相关的 Dag 以及可以从 Google sheets 中提取数据的 Sheetload。

随着所有不必要的东西消失，我为 [Fitbit](https://gitlab.com/snowboard-analysis/analytics/blob/master/extract/fitbit_load.py) 和 [Trace](https://gitlab.com/snowboard-analysis/analytics/blob/master/extract/trace_load.py) 创建了我的提取，以及每个的[Dag](https://gitlab.com/snowboard-analysis/analytics/tree/master/dags/extract)。我还必须为[数据映像](https://gitlab.com/snowboard-analysis/data-image/tree/master/data_image)更新`requirements.txt`文件，以便添加 Trace 和 Fitbit 包。

# 建立 GKE

如果您还没有 Google Cloud 帐户，请创建一个。如果你创建一个新的，你会得到 300 美元的信用，这是免费试用这个项目的完美选择。创建好之后，在[控制台](https://console.cloud.google.com/apis/api/container.googleapis.com/overview)中启用 GKE API。

然后，转到[创建一个角色为`Kubernetes Engine Developer`的服务帐户](https://console.cloud.google.com/iam-admin/serviceaccounts)。当询问密钥时，创建一个并下载它。

一旦集群设置好了(见上面通过 Gitlab 设置)，就该在本地设置 kubernetes 来使用我们创建的集群了。一个简单的方法是:

```
gcloud auth activate-service-account --key-file </path/to/downloaded/key.json>
gcloud container clusters get-credentials data-ops --region us-west1-a --project <name-of-your-project>
```

现在，为了上传必要的秘密，创建一个`secrets.yaml`文件(填入您自己的值):

```
apiVersion: v1
kind: Secret
metadata:
  name: airflow
type: Opaque
stringData:
    SNOWFLAKE_ACCOUNT: ""
    SNOWFLAKE_PASSWORD: ""
    SNOWFLAKE_USER: ""
    SNOWFLAKE_LOAD_DATABASE: "RAW"
    SNOWFLAKE_LOAD_PASSWORD: ""
    SNOWFLAKE_LOAD_ROLE: "LOADER"
    SNOWFLAKE_LOAD_USER: ""
    SNOWFLAKE_LOAD_WAREHOUSE: "LOADING"
    SNOWFLAKE_TRANSFORM_DATABASE: "ANALYTICS"
    SNOWFLAKE_TRANSFORM_SCHEMA: "ANALYTICS"
    SNOWFLAKE_TRANSFORM_PASSWORD: ""
    SNOWFLAKE_TRANSFORM_ROLE: "TRANSFORMER"
    SNOWFLAKE_TRANSFORM_USER: ""
    SNOWFLAKE_TRANSFORM_WAREHOUSE: "TRANSFORMING_S"
    SNOWFLAKE_PERMISSION_USER: ""
    SNOWFLAKE_PERMISSION_PASSWORD: ""
    SNOWFLAKE_PERMISSION_ROLE: "PERMISSION_BOT"
    SNOWFLAKE_PERMISSION_DATABASE: "SNOWFLAKE"
    SNOWFLAKE_PERMISSION_WAREHOUSE: "ADMIN"
    TRACE_CLIENT_KEY: ""
    TRACE_CLIENT_SECRET: ""
    TRACE_OAUTH_TOKEN: ""
    TRACE_OAUTH_TOKEN_SECRET: ""
    FITBIT_CLIENT_ID: ""
    FITBIT_CLIENT_SECRET: ""
    FITBIT_ACCESS_TOKEN: ""
    FITBIT_REFRESH_TOKEN: ""
    AIRFLOW__CORE__SQL_ALCHEMY_CONN: ""
    AIRFLOW__CORE__FERNET_KEY: ""
    NAMESPACE: "default"
    cloudsql-credentials: |-
      {
      "type": "service_account",
      "project_id": "",
      "private_key_id": "",
      "private_key": "",
      "client_email": "",
      "client_id": "",
      "auth_uri": "",
      "token_uri": "": "",
      "client_x509_cert_url": ""
      }
```

我把所有东西都放在`stringData`而不是`data`里，只是为了更容易查看和编辑。然而，对于 prod 环境，您可能希望使用`data`并自己进行 base 64 编码。`cloudsql-credentials`是您从 Google 下载的服务帐户凭证。像这样嵌入 json 时，缩进很重要！

要上传秘密，只需运行`kubectl apply -f secrets.yaml`。当在本地运行 airflow 进行测试时，名称空间被设置为`testing`。为了创建名称空间，运行`kubectl create namespace testing`并上传机密，运行`kubectl apply --namespace testing -f ./secrets.yaml`。

# 设置雪花

前往 https://www.snowflake.com/开始 30 天的免费试用。当您创建一个帐户时，您也将创建一个具有`SYSADMIN`、`ACCOUNTADMIN`和`SECURITYADMIN`角色的主用户。尽管 Gitlab 数据团队使用各种用户、角色、数据库和仓库，但我简化了其中的大部分，只使用一个用户和几个不同的角色。关于权限，我还有相当多的东西要学习，但是在使用了一段时间之后，我能够让事情运行起来。

repo 有一个`[roles.yaml](https://gitlab.com/gitlab-data/analytics/blob/master/load/snowflake/roles.yml)`文件，描述每个角色、用户、仓库、数据库和模式以及它们的权限。他们有一个 DAG，使用 [Meltano permissions](https://meltano.com/docs/command-line-interface.html#permissions) 命令来生成设置权限所需的 SQL。目前，这只能在干模式下运行，但我想目标是 Meltano 应该能够处理所有权限的设置。我不得不运行它，复制粘贴 SQL，并做一些修改，让事情顺利进行。因为 DAG 使用`PERMISSION_BOT`角色来运行，所以有一点先有鸡还是先有蛋的问题，所以我必须首先创建它并给它适当的权限。如果使用单个用户和角色，我可以将事情简化得多，设置起来也会容易得多，但我想更好地了解 Gitlab 是如何做到这一点的。

# 设置气流

大部分气流设置在[气流图像](https://gitlab.com/snowboard-analysis/data-image/tree/master/airflow_image)中完成。我对配置做了一些更改，主要是为了减少活动连接的数量，我还对图像做了一些更改，使它们变得更小。我还必须更改部署清单中的硬编码项目名称，以匹配我的项目。

为简单起见，我只使用了 SaaS Postgres 数据库的气流数据。我用了 https://www.elephantsql.com/的[但是任何公共 Postgres db 都可以。需要记住的一点是，在任何给定的时间点，您都有 10 到 20 个活动连接(来自 web 服务器、调度器和工作器)。在那之后，我只是如上所述在我的 Kubernetes 秘密中添加了`AIRFLOW__CORE__SQL_ALCHEMY_CONN`
和`AIRFLOW__CORE__FERNET_KEY`。](https://www.elephantsql.com/)

要部署，只需:

```
cd airflow_image/manifests
apply -f ./ingress.yml
apply -f ./persistent_volume.yaml
apply -f ./services.yml
apply -f ./deployment.yaml
```

这将创建并启动所有必要的 Kubernetes 组件来部署气流，并公开 web 服务器。要在本地访问它，运行`kubectl port-forward deployment/airflow-deployment 1234:8080`，然后你可以在`localhost:1234`打开它。

# 建立 DBT

DBT 主要作为 DAG 运行。有一个 docker 映像，它捆绑了运行 DBT 命令所需的所有软件包，该映像用于在 DAG 中作为任务运行不同的命令。DBT 项目结构是分析报告的子目录。我在这里做的主要事情是创建我自己的 Fitbit 和 Trace dbt [模型](https://gitlab.com/snowboard-analysis/analytics/tree/master/transform/snowflake-dbt/models)，它们执行以下操作:

*   创建标准的 snake case 列名
*   将单位从追踪制转换为英制
*   每天汇总 Fitbit 睡眠数据
*   加入每天的所有 Fitbit 测量

DBT 的一个非常酷的功能是能够测试来自你的模型的数据。这有助于确保没有重复条目、空值等。我仍然没有玩这个，但以后会得到它。

在生产中，DBT `profiles.yml`文件是使用来自 secrets 的 env 变量生成的。然而，对于本地测试，您应该创建自己的`~/.dbt/profiles.yml`文件，看起来像这样:

```
gitlab-snowflake:
  target: dev
  outputs:
    dev:
      type: snowflake
      threads: 8
      account: 
      user: 
      password: 
      role: TRANSFORMER
      database: ANALYTICS
      warehouse: TRANSFORMING_S
      schema: ANALYTICS
```

# 测试气流 Dag 和 DBT

主回购有一个`docker-compose.yml`和一个`Makefile`，这使得在本地测试 Dag 和 DBT 模型变得容易。然而，这仍然需要 Kubernetes 集群来运行气流任务，并且还需要`testing`名称空间中的秘密(见上文)。

每当您运行 Airflow DAGs 时，DAG 定义将从您的本地 repo 中提取，但是 Kubernetes pod 将使用指定的`GIT_BRANCH`从注册表中提取最新的图像，并从托管 repo 中提取最新的代码，因此您对提取代码的更改将需要首先被推送。

要使用提供的 docker-compose 和 Makefile，您需要设置几个环境变量:

```
export DBT_PROFILE_PATH=~/.dbt/profiles.yml
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/legacy_credentials/<service_account_email>/adc.json
export KUBECONFIG=~/.kube/config
export GIT_BRANCH=<your_test_branch>
```

您可以在这里阅读一些关于如何在本地[测试的内容，但它主要涉及到使用`make init-airflow`在本地 postgres 容器上设置 Airflow db，使用`make airflow`进入 Airflow 容器并运行 Airflow 命令，和/或使用`make dbt-image`进入 DBT 容器并运行 dbt 命令。](https://about.gitlab.com/handbook/business-ops/data-team/data-infrastructure/#in-merge-requests)

# 结论和经验教训

现在，你可能想知道“为什么”？这不是矫枉过正吗？嗯，是的，是的，但我这样做的原因是为了熟悉 Gitlab 管道。这是一个很好的方式来了解 Kubernetes，DBT，雪花和使用 KubernetesPodOperator on Airflow 运行任务。虽然我对其中的每一个工具都只是略知皮毛，但是我现在已经有了一个更好的基础来继续学习这些不同的工具。在我看来，通过做来学习比仅仅通过阅读要容易得多。

我从这个练习中得到了一些教训/收获:

*   Kubernetes 和它上面的所有机器都按 UTC 时间运行。尽管这对于大多数服务器来说很常见，但是在 Kubernetes 中调试某些问题有点困难。我遇到的一个问题是`datetime.today().timestamp()`会根据机器设定的时区给出不同的结果。当我在本地测试我的提取时，它工作得很好，但是当在 Kubernetes 上运行时，它不能像预期的那样工作。在添加了一堆调试日志之后，我意识到这是因为 Trace 必须使用每个度假胜地的时区作为`date`的时区，所以当我过滤时，我需要告诉它时区在`MST`。
*   尽管在 Pandas 中以编程方式编写转换可能更容易，但使用 DBT 使分析人员更容易理解。它还允许您通过组合各种其他已定义的模型并使用宏来构建非常复杂的模型。DBT 非常强大，我有很多关于最佳实践的东西要学习。有许多不同的方法来配置和构建您的模型，有不同的权衡。我希望看到一个使用 dbt 的 Emacs 模式，但是看起来我可能必须自己创建它。
*   如果您需要在 KubernetesPodOperator 上运行的 Airflow 任务中更改密码，请参见下面的文章:[从 Kubernetes Pod 更新密码](/@aiguofer/updating-secrets-from-a-kubernetes-pod-f3c7df51770d)。