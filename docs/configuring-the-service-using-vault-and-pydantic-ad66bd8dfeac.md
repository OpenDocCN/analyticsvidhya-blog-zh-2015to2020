# Sitri = Vault+Pydantic:具有清晰结构和验证的配置

> 原文：<https://medium.com/analytics-vidhya/configuring-the-service-using-vault-and-pydantic-ad66bd8dfeac?source=collection_archive---------18----------------------->

![](img/713ad66260c91a1516e4e6d078db3c1a.png)

# 介绍

继续，立即，以防万一

在这篇文章中，我将讨论在 [Sitri](https://github.com/LemegetonX/sitri) 的赞助下，使用 Vault (KV 和到目前为止只有第一个版本，即没有版本秘密)和 Pydantic(设置)为您的服务进行配置。

因此，假设我们有一个 *superapp* 服务，在 Vault 中设置了配置，并使用 AppRole 进行身份验证，我们将像这样设置它(我将把访问秘密引擎和秘密本身的策略设置留在幕后，因为这非常简单，本文不涉及这一点):

```
Key                        Value
---                        -----
bind_secret_id             true
local_secret_ids           false
policies                   [superapp_service]
secret_id_bound_cidrs      <nil>
secret_id_num_uses         0
secret_id_ttl              0s
token_bound_cidrs          []
token_explicit_max_ttl     0s
token_max_ttl              30m
token_no_default_policy    false
token_num_uses             50
token_period               0s
token_policies             [superapp_service]
token_ttl                  20m
token_type                 default
```

*注意:*自然，如果你有机会，应用进入生产模式，那么 *secret_id_ttl* 最好不要无限，设置 0 秒。

*Superapp* 需要配置数据库连接，连接到 kafka 和 [faust](http://faust.readthedocs.io) 配置集群工人。

最后，我们应该有这样的测试项目结构:

```
super_app
├── config
│ ├── __init__.py
│ ├── provider_config.py
│ ├── faust_settings.py
│ ├── app_settings.py
│ ├── kafka_settings.py
│ └── database_settings.py
├── __init__.py
└── main.py
```

# 烘焙 Sitri 提供商

基本库文档中有一个简单的示例，通过 vault provider 进行配置，但是，它没有涵盖所有功能，如果您的应用程序配置足够简单，它可能会很有用。

因此，首先，让我们配置存储库提供程序:

在此代码中，我们使用系统提供程序从环境中获取变量，以配置与 vault 的连接，即必须首先导出以下变量:

```
export SUPERAPP_ENV=dev
export SUPERAPP_APP_NAME=superapp
export SUPERAPP_VAULT_API=[https://your-vault-host.domain](https://your-vault-host.domain)
export SUPERAPP_ROLE_ID=<YOUR_ROLE_ID>
export SUPERAPP_SECRET_ID=<YOUR_SECRET_ID>
```

该示例假设针对特定环境的 secrets 的 base *mount_point* 将包含应用程序名称和环境名称，这就是我们导出 *SUPERAPP_ENV* 的原因。稍后我们将在设置类中定义应用程序各个部分的秘密路径，因此我们在*secret _ path*vault provider 参数中将其留空。

# 设置类别

## 数据库设置—数据库连接

要创建设置类，我们必须使用 VaultKVSettings 作为基类。

如您所见，数据库连接的配置数据非常简单。默认情况下，这个类会查看秘密 *superapp/dev/db* ，正如我们在 *Config* 类中指定的那样。乍一看，这些是简单的*止痛药。字段*，但是它们都有一个额外的参数 *vault_secret_key* —当 secret 中的密钥与我们类中的 pydantic 字段的名称(别名)不匹配时就需要这个参数，如果没有指定 *vault_secret_key* ，提供者将通过字段别名来搜索密钥。

例如，在我们的 *superapp* 中，假设 *superapp/dev/db* secret 具有“password”和“username”密钥，但是为了方便和简洁，我们希望将后者放在“user”字段中。

让我们把下面的数据放在上面的秘密里:

```
{
  "host": "testhost",
  "password": "testpassword",
  "port": "1234",
  "username": "testuser"
}
```

现在，如果我们运行这段代码，我们将使用我们的 *DBSettings* 类从 vault 中获取我们的秘密:

## KafkaSettings —与经纪人的联系

在这种情况下，让我们假设对于我们服务的不同环境有一个 kafka 实例，因此秘密沿着路径 *superapp/common/kafka* 存储:

*注*:您也可以在字段级设置 *secret_path* 或/和 *mount_point* ，以便提供商从不同的 secret 请求特定值(如果需要)。这里引用了[文档](https://sitri.readthedocs.io/en/latest/advanced_usage.html#vault-settings-configurators)中关于秘密路径和挂载点优先级的内容:

> 秘密路径优先级:
> 1。保险库 _ 秘密 _ 路径(字段参数)
> 2。默认 _ 秘密 _ 路径(配置类字段)
> 3。secret_path(提供者初始化可选参数)
> 
> 挂载点优先级:
> 1。vault _ mount _ point(Field arg)
> 2。默认挂载点(配置类字段)
> 3。mount_point(提供者初始化可选参数)

```
{
  "auth_data": "{\"password\": \"testpassword\", \"username\": \"testuser\"}",
  "auth_mechanism": "SASL_PLAINTEXT",
  "brokers": "kafka://test"
}
```

或者

```
{
    "auth_data":
    {
        "password": "testpassword",
        "username": "testuser"
    },
    "brokers": "kafka://test",
    "auth_mechanism": "SASL_PLAINTEXT"
}
```

*注:* *VaultKVSettings* 既可以理解 json，也可以理解 Dict 本身。

因此，这类设置将能够像这样从秘密中收集数据:

```
{
    "auth_data":
    {
        "password": "testpassword",
        "username": "testuser"
    },
    "brokers": "kafka://test",
    "auth_mechanism": "SASL_PLAINTEXT"
}
```

## faustSettings —全局配置 Faust 和单个代理

秘密 *superapp/dev/faust* :

```
{
  "agent_concurrency": "5",
  "app_name": "superapp-workers",
  "partitions_count": "10"
}
```

如果我们的秘密如上所述编写，那么各个代理将从默认字段中获取关于其主题中的分区数量和并发性的信息。

```
{
  "agents": None,
  "app_name": "superapp-workers",
  "default_concurrency": 5,
  "default_partitions_count": 10
}
```

然而，在 *AgentConfig* 模型的帮助下，我们可以为特定代理设置单独的值。例如，如果代理 *X* 有 5 个分区，并发数为 2，那么我们可以更改我们的秘密，这样关于这个代理的信息就在“代理”字段中。

秘密 *superapp/dev/faust* :

```
{
  "agent_concurrency": "5",
  "agents_specification": {
    "X": {
      "concurrency": "2",
      "partitions": "5"
    }
  },
  "app_name": "superapp-workers",
  "partitions_count": "10"
}
```

如果我们现在初始化设置，除了默认字段之外，我们还将收到特定代理的配置:

```
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
}
```

## 将设置类合并到单个配置模型中

为了使我们的配置更加方便使用，让我们通过应用 *default_factory* 将所有的设置类合并到一个模型中:

现在，让我们转到 *main.py* 文件，测试我们的应用程序的完整配置数据的收集:

我们得到应用程序配置的输出:

```
db=DBSettings(user='testuser', password='testpassword', host='testhost', port=1234)

faust=FaustSettings(app_name='superapp-workers', default_partitions_count=10, default_concurrency=5, agents={'X': AgentConfig(partitions=5, concurrency=2)})

kafka=KafkaSettings(auth_mechanism='SASL_PLAINTEXT', brokers='kafka://test', auth_data={'password': 'testpassword', 'username': 'testuser'}){
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
        "auth_mechanism": "SASL_PLAINTEXT"
    }
}
```

***幸福、快乐、喜悦！***

# 编后记

如您所见，使用 [Sitri](http://sitri.readthedocs.io) 进行配置非常简单，之后我们会得到一个清晰的配置方案，其中包含值所需的数据类型，即使它们默认存储在 vault 中的字符串中。

写下关于库、代码或一般印象的评论。我将很高兴得到任何反馈！

附注[我已经将文章中的代码上传到 github](https://github.com/Egnod/article_sitri_vault_pydantic)