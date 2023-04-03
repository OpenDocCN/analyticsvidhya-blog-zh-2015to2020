# Sitri = Vault + Pydantic:传奇的延续，局部发展。

> 原文：<https://medium.com/analytics-vidhya/sitri-vault-pydantic-continuation-of-the-saga-local-development-6ccd11da17e0?source=collection_archive---------20----------------------->

![](img/a815c6e716de59ac5f9ffe0e8d7acf2e.png)

# 背景

在[上一篇文章](https://egnod.medium.com/configuring-the-service-using-vault-and-pydantic-ad66bd8dfeac)中，我写了如何使用 [Sitri](https://sitri.readthedocs.io/en/latest/) 配置您的应用程序，然而，我忽略了本地开发的要点，因为您会同意在本地部署一个 Vault 并不方便，并且在一个公共 Vault 中存储一个本地配置更加不方便，特别是如果几个人在一个项目上工作的话。

在 [Sitri](https://sitri.readthedocs.io/en/latest/) 中，这个问题解决得非常简单——使用本地模式设置类，也就是说，你甚至不需要重写任何东西或复制代码，本地模式的结构 json 文件将几乎完全重复秘密的结构。

那么，现在，让我们给我们的[项目](https://github.com/Egnod/article_sitri_vault_pydantic)添加几行代码+如果您的项目在 docker-compose 中本地运行，我将向您展示如何使用所有这些代码…

# 准备代码

首先，让我们同意当 *ENV* = "local ":)时 local_mode 为真

接下来，我建议稍微编辑一下我们的 *provider_config.py* 并创建一个 *BaseConfig* 类，我们将从那里继承我们的 *Config* 设置类。我们这样做是为了不重复代码，也就是说，设置类本身将只包含特定于它们的内容。

关于 *local_provider_args* 在这个字段中，我们指定了用于创建 *JsonConfigProvider* 实例的参数，它们将被验证，并且这个字典必须匹配模式，所以不要担心——这不是什么肮脏的把戏。然而，如果您想自己创建一个本地提供者的实例，那么您只需将它放在可选的 *local_provider* 字段中。

现在，我们可以很容易地从基类继承配置类。例如，用于连接到 Kafka 的设置类如下所示:

如您所见，所需的更改很少。我们指定通用配置的结构保存在我们的 json 文件中。现在，让我们为本地配置编写这个 json 文件:

```
{
    "db":
    {
        "host": "testhost",
        "password": "testpassword",
        "port": 1234,
        "user": "testuser"
    },
    "faust":
    {
        "agents":
        {
            "X":
            {
                "concurrency": 2,
                "partitions": 5
            }
        },
        "app_name": "superapp-workers",
        "default_concurrency": 5,
        "default_partitions_count": 10
    },
    "kafka":
    {
        "auth_data":
        {
            "password": "testpassword",
            "username": "testuser"
        },
        "brokers": "kafka://test",
        "mechanism": "SASL_PLAINTEXT"
    }
}
```

…嗯，或者直接从上一篇文章的末尾复制粘贴。如你所见，这里的一切都很简单。为了我们的进一步研究，将项目根中的 *main.py* 重命名为 *__main__。这样你就可以用 docker-compose 命令运行这个包了。*

# 将应用程序放入容器并享受构建过程

我们应该做的第一件事是写一个小 docker 文件:

在这里，我们只安装依赖项，就是这样，因为它是本地开发，我们不复制项目代码。

接下来，我们需要一个包含本地模式所需变量的 env 文件:

如您所见，没有多余的东西，Vault 不需要配置信息，因为在本地模式下，应用程序甚至不会尝试“敲”Vault。

我们需要写的最后一件事是 docker-compose.yml 文件本身:

这里一切都很简单。我们将 json 文件放在根目录中，就像上面在容器的环境变量中写的那样。

现在，发射:

```
docker-compose upCreating article_sitri_vault_pydantic_superapp_1 ... done
Attaching to article_sitri_vault_pydantic_superapp_1superapp_1  | db=DBSettings(user='testuser', password='testpassword', host='testhost', port=1234) faust=FaustSettings(app_name='superapp-workers', default_partitions_count=10, default_concurrency=5, agents={'X': AgentConfig(partitions=5, concurrency=2)}) kafka=KafkaSettings(mechanism='SASL_PLAINTEXT', brokers='kafka://test', auth_data={'password': 'testpassword', 'username': 'testuser'})superapp_1  | {'db': {'user': 'testuser', 'password': 'testpassword', 'host': 'testhost', 'port': 1234}, 'faust': {'app_name': 'superapp-workers', 'default_partitions_count': 10, 'default_concurrency': 5, 'agents': {'X': {'partitions': 5, 'concurrency': 2}}}, 'kafka': {'mechanism': 'SASL_PLAINTEXT', 'brokers': 'kafka://test', 'auth_data': {'password': 'testpassword', 'username': 'testuser'}}}
```

正如您所看到的，一切都成功地启动了，来自我们 json 文件的信息成功地通过了所有检查，并成为应用程序的本地版本 yuhhu 的设置！

我把这个“延续”的代码放在了存储库的一个单独的分支中，所以你可以看看它在更改后是什么样子的:[分支](https://github.com/Egnod/article_sitri_vault_pydantic/tree/local_mode_example)