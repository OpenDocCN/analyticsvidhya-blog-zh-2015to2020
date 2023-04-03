# 从 Kubernetes 豆荚更新秘密

> 原文：<https://medium.com/analytics-vidhya/updating-secrets-from-a-kubernetes-pod-f3c7df51770d?source=collection_archive---------6----------------------->

![](img/9352fe84daeff02d0e38158d4b48d55c.png)

由 [Safar Safarov](https://unsplash.com/@codestorm?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

有时，您可能想要更新 Kubernetes Pod 中的秘密。例如，如果您有 OAuth 应用程序的访问令牌和刷新令牌，那么您可能需要在每次刷新它们时更新它们。我确信有几种不同的方法可以实现这一点，并且总是可以选择将它们存储在 DB 中。就我而言，我需要为 Fitbit 更新我的令牌，并希望将它们作为秘密保存在 Kubernetes 中。在网上看了一些东西，找到了`kubectl patch`命令，好像就是我想要的。完整的命令应该是这样的:

```
kubectl patch secret <secret> -p='{"data":{"<key>": "<base64_encoded_val>"}}
```

我想用 Python 来做这件事，所以我写了以下函数，它们在我的机器上运行得很好:

```
from base64 import b64encode
from subprocess import rundef b64encodestr(string):
    return b64encode(string.encode("utf-8")).decode()def update_secrets(secret, key, val):
    b64val = b64encodestr(val)
    cmd = f"""kubectl patch secret {secret} -p='{{"data":{{"{key}": "{b64val}"}}}}'"""
    return run(cmd, shell=True)
```

然后我使用 KubernetesPodOperator 创建了一个简单的 DAG，它会通过每分钟用当前时间戳更新`TEST_SECRET`来更新我的`airflow`秘密。要使用这种方法，您首先需要确保在您正在使用的映像中安装了`kubectl`。当我第一次运行它时，我得到了以下错误:

```
Error from server (Forbidden): secrets "airflow" is forbidden: User "system:serviceaccount:default:default" cannot get resource "secrets" in API group "" in the namespace "default"
```

读了一点，我发现这是由于角色和权限。为了修复它，我首先需要创建一个允许更新秘密的角色。我用下面的命令做到了这一点(注意，您将需要从 GCP 分配的`Kubernetes Engine Admin`角色来运行这些命令):

```
kubectl create role update-secrets --verb=get,patch --resource=secrets
```

一旦我创建了角色，我必须将它分配给我在`default`名称空间上的`default`服务帐户(这是 KubernetesPodOperator 创建的 pods 中的默认设置)。您可以通过以下方式实现这一点:

```
kubectl create rolebinding --role=update-secrets default-update-secrets --serviceaccount=default:default
```

在这之后，一切都在按预期工作！现在，我可以在每天提取数据时更新我的访问权限并刷新我的 fitbit 帐户的令牌！

注意:如果你只是使用 Python，你也可以使用`kubernetes` python 库和`patch_namespaced_secret`函数。出于本文的目的，我决定使用最“便携”的选项。