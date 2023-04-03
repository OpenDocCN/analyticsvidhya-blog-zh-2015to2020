# 使用 Python 中的 Azure Key Vault 保护您的秘密

> 原文：<https://medium.com/analytics-vidhya/keep-your-secrets-safe-with-azure-key-vault-in-python-9848be3230db?source=collection_archive---------4----------------------->

![](img/e4c522494f878878d2ac958202f35d99.png)

使用 Azure Key Vault 保护您的秘密。

当你开发应用程序时，你经常会遇到这样的情况，你必须将你的工作与其他应用程序和资源连接起来。因为您只想快速检查连接方法是否有效，所以您在代码中将用户名和密码作为不变的字符串写下来，运行代码。瞧！起作用了！现在是喝那杯当之无愧的咖啡的时候了…

…在啜饮您的 vic tourisy 棕色液体时，您提交您的更改并将代码推送到您与各种同事共享的应用程序存储库。就是这样，凭证可以被多人访问，并且那些人可以(不希望地)访问他们的计算机。

现在，让我们确保这不会发生在你身上！

我假设你有一个有效的 Azure 订阅，安装了 Python 2.7 或 3.5+和 Azure CLI。

# 创建和配置 Azure 资源

让我们从创建 Azure 资源组和密钥库开始:

```
$ az login$ az group create --name <keyVaultGroup> -l westeurope$ az keyvault create --name <myUniqueKeyVaultName> -g <keyVaultGroup>
```

检索并记住密钥库`properties.vaultUri`:

```
$ az keyvault show -n <myUniqueKeyVaultName> --query "properties.vaultUri" -o json
```

密钥库 URL 应该是这样的:`https://myuniquekeyvaultname.vault.azure.net/`

现在创建一个服务主体来管理访问策略:

```
$ az ad sp create-for-rbac --name <http://my-key-vault-principal-name> --sdk-auth
```

注意输出，特别是`clientId`、`clientSecret`和`tenantId`，它应该是这样的:

```
{
  "clientId": "d55c654c-22df-4641-91a9-3e8b567d6253",
  "clientSecret": "1pXue~dOK6TTCf1NVTBcMaFrQwnIWjS_tO",
  "subscriptionId": "d410egf1-7e00-45x3-beab-081531f878ed",
  "tenantId": "45c8b1a0-8c74-43f9-9fd5-110f80d9a6f9",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}
```

创建允许服务主体访问密钥库的访问策略。**注意**上一步输出的`--spn` ID 是`clientId`:

```
$ az keyvault set-policy -n <myUniqueKeyVaultName> --spn <the-previous-clientId> --secret-permissions delete get list set --key-permissions create decrypt delete encrypt get list unwrapKey wrapKey
```

# 设置环境变量

我们稍后将在 Python 中使用的 Azure Identity 客户端库将查找环境服务主体变量，以向密钥库认证自身。

## Linux 操作系统

打开终端，编辑`$HOME`目录下的`.profile`文件:

```
$ cd ~ && nano .profile
```

现在，将服务主体的细节添加到文件的末尾，并保存它:

```
AZURE_CLIENT_ID="<the-previous-clientId>"
AZURE_CLIENT_SECRET="<the-previous-clientSecret>"
AZURE_TENANT_ID="<the-previous-tenantId>"
VAULT_URL="<the-previous-properties.vaultUri>"
```

## Windows 操作系统

以管理员权限打开 CMD，并将服务主体的细节设置为环境变量:

```
$ SETX AZURE_CLIENT_ID "<the-previous-clientId>"
$ SETX AZURE_CLIENT_SECRET "<the-previous-clientSecret>"
$ SETX AZURE_TENANT_ID "<the-previous-tenantId>"
$ SETX VAULT_URL "<the-previous-properties.vaultUri>"
```

# Python 实现

## 安装需求

安装必要的软件包:

```
pip install azure.keyvault
pip install azure.identity
```

## 设置秘密

以下示例将密码添加到密钥库中:

```
import osfrom azure.identity import EnvironmentCredential
from azure.keyvault.secrets import SecretClientVAULT_URL = os.environ["VAULT_URL"]credential = EnvironmentCredential()
client = SecretClient(vault_url=VAULT_URL, credential=credential)client.set_secret(
    "<my-password-reference>",
    "<the-actual-password>",
)
```

## 获取秘密

以下示例使用“设置密码”步骤中的客户端从密钥库中获取密码:

```
password = client.get_secret("<my-password-reference>").value
```

## 删除秘密

以下示例从密钥库中删除一个密码，使用“设置密码”步骤中的客户端:

```
client.begin_delete_secret("<my-password-reference>")
```

就是这样，现在您可以将任何代码推送到任何(公共)存储库，而没有暴露凭据的风险，您的秘密是安全的！