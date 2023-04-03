# 用于创建简单服务的 Azure 函数(无服务器概念)

> 原文：<https://medium.com/analytics-vidhya/azure-function-serverless-concept-for-creating-a-simple-service-b1bc5a110c9?source=collection_archive---------15----------------------->

无服务器的概念并不新鲜，但它的流行才开始不久。2014 年，无服务器与 Kubernetes 同时出现。对于不同的服务提供商，无服务器的概念有不同的名称，在 Azure 或 GCP 中称为 Function (fx ),而在 AWS 中称为 know by lambda，在幕后它的工作原理是相同的。

在这个演示中，我将使用 Azure Function (fx ),但我将使用 AWS 来部署过去系列[中的服务。让我们从简单的“你好，世界！”Azure 模板。在本地机器上创建一个目录。一旦有了要使用的目录，就可以通过以下命令创建 Azure 函数(fx)。因为我们使用 python 作为我们的服务，所以 init-worker 是 python。](/delvify/bert-rest-inference-from-the-fine-tuned-model-499997b32851?source=friends_link&sk=922b4670ed9f9673f2e47c9c972f8484)

```
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ func init --worker-runtime python
Found Python version 3.6.8 (python).
Writing .gitignore
Writing host.json
Writing local.settings.json
Writing /home/jugs/Desktop/azurefxDemo/.vscode/extensions.json
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ 
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ tree
.
├── host.json
├── local.settings.json
└── requirements.txt0 directories, 3 files
```

创建了以下配置文件，要了解文件的细节可以参考 Azure Function 文档页面，现在，我将跳过这些配置文件后面的细节。

所以现在我们可以用函数命令`func new --name exampledemo --template “Http trigger`从模板或“Hello World”创建一个最小的应用程序。这将创建一个具有给定名称和起始应用程序文件的新目录。下图展示了我们所看到的，

```
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ func new --name exampledemo --template "Http trigger"
Select a template: Http trigger
Function name: [HttpTrigger] Writing /home/jugs/Desktop/azurefxDemo/exampledemo/__init__.py
Writing /home/jugs/Desktop/azurefxDemo/exampledemo/function.json
The function "exampledemo" was created successfully from the "Http trigger" template.
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ tree
.
├── exampledemo
│   ├── function.json
│   └── __init__.py
├── host.json
├── local.settings.json
└── requirements.txt1 directory, 5 files
```

名为`function.json`的文件是配置文件，`__init__.py`是我们要使用的应用文件。只需这么多，您就可以使用函数 run 命令`func start`运行本地主机 web 服务。

```
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ func start
Found Python version 3.6.8 (python).%%%%%%
                 %%%%%%
            @   %%%%%%    @
          @@   %%%%%%      @@
       @@@    %%%%%%%%%%%    @@@
     @@      %%%%%%%%%%        @@
       @@         %%%%       @@
         @@      %%%       @@
           @@    %%      @@
                %%
                %Azure Functions Core Tools (2.7.1948 Commit hash: 29a0626ded3ae99c4111f66763f27bb9fb564103)
Function Runtime Version: 2.0.12888.0
...................
...................[2/5/20 3:48:08 AM] Job host started
Hosting environment: Production
Content root path: /home/jugs/Desktop/azurefxDemo
Now listening on: [http://0.0.0.0:7071](http://0.0.0.0:7071)
Application started. Press Ctrl+C to shut down.Http Functions:exampledemo: [GET,POST] [http://localhost:7071/api/exampledemo](http://localhost:7071/api/exampledemo)[2/5/20 3:48:08 AM]  INFO: Starting Azure Functions Python Worker.
```

您现在可以卷曲或使用您的浏览器来检查服务。这就是模板的样子。

```
(myvenv) jugs@jugs:~$ curl [http://localhost:7071/api/exampledemo](http://localhost:7071/api/exampledemo)
Please pass a name on the query string or in the request body(myvenv) jugs@jugs:~$ 
(myvenv) jugs@jugs:~$ 
(myvenv) jugs@jugs:~$ curl [http://localhost:7071/api/exampledemo?name=Azure%20Function](http://localhost:7071/api/exampledemo?name=Azure%20Function)
Hello Azure Function!
```

到目前为止，我们都在本地主机中，让我们开始在 Azure web 服务中创建一个 Azure 函数，然后在 Azure 中推送和构建我们的程序。

1.  从你的浏览器使用你的 Azure 登录凭证登录 Azure `$ az login`。
2.  创建一个离你最近的资源组`$ az group create --name myResGroup --location westeurope`
3.  创建存储帐户`$ az storage account create --name myStorageName --location westeurope --resource-group --myResGroup --sku standard_LRS`，您可以根据您的需求从列表中选择 SKU。
4.  现在，一旦我们有了所有的成分，你就可以创建一个函数`$ az functionapp create resource-group myResGroup --consumption-plan-location westeurope --name myAppName --storage-account myStorageName --runtime python`，对于`--runtime`，我们应该指定语言作为我们的参数。

您可以从 Azure 门户查看在资源组中创建的资源。最低要求应该是这样的。

![](img/274ddba5cc5c875224ff7574a2306d1c.png)

图:Azure 功能资源组

最后，我们可以将代码从本地推送到我们的 Azure 函数并构建`$ az func azure functionapp publish myAppName --build remote`，这个命令会将你所有的代码和数据从本地推送到你的 Azure 资源组。

```
...............
func azure functionapp publish simpleFuncClass --build remote
Getting site publishing info...
Creating archive for current directory...
Performing remote build for functions project.
Deleting the old .python_packages directory
Uploading 132.48 KB [#############################################################################]
Remote build in progress, please wait...
Updating submodules.
Preparing deployment for commit id 'a9ac5db900'.
Repository path is /tmp/zipdeploy/extracted
Running oryx build...
Command: oryx build /tmp/zipdeploy/extracted -o /home/site/wwwroot --platform python --platform-version 3.6 -p packagedir=.python_packages/lib/python3.6/site-packages
Build orchestrated by Microsoft Oryx, [https://github.com/Microsoft/Oryx](https://github.com/Microsoft/Oryx)
You can report issues at [https://github.com/Microsoft/Oryx/issues](https://github.com/Microsoft/Oryx/issues)Oryx Version      : 0.2.20191018.1, Commit: b56054ad55c50d02d333db35752b7f0aee706ad5, ReleaseTagName: 20191018.1
Build Operation ID: |Ta78ExE4dr0=.45926069_
Repository Commit : a9ac5db9005d4a5d9bf926f173ee644a..................
..................
Number of files 12066
Number of fragments 925
Number of symbolic links  0
Number of device nodes 0
Number of fifo nodes 0
Number of socket nodes 0
Number of directories 2636
Number of ids (unique uids + gids) 1
Number of uids 1
 root (0)
Number of gids 1
 root (0)
Uploading built content /home/site/deployments/functionappartifact.squashfs -> [https://funcjugsstorage01.blob.core.windows.net/scm-releases/scm-latest-simpleFuncClass.zip?sv=2014-02-14&sr=b&sig=a6aGacuN2hIr9o2ThF6NYn%2Fj7uDyc5nSz4u6LNCsaBE%3D&se=2030-01-27T04%3A24%3A44Z&sp=rw](https://funcjugsstorage01.blob.core.windows.net/scm-releases/scm-latest-simpleFuncClass.zip?sv=2014-02-14&sr=b&sig=a6aGacuN2hIr9o2ThF6NYn%2Fj7uDyc5nSz4u6LNCsaBE%3D&se=2030-01-27T04%3A24%3A44Z&sp=rw)
Resetting all workers for simplefuncclass.azurewebsites.net
Deployment successful.
Remote build succeeded!
Syncing triggers...
Functions in simpleFuncClass:
    HttpExample - [httpTrigger]
        Invoke url: [https://simplefuncclass.azurewebsites.net/api/httpexample?code=c0U7ThcnusamBpQRscAulAhC488PlKhVbnAbi2pdxVnqqjyvajlQ==](https://simplefuncclass.azurewebsites.net/api/httpexample?code=c0U7ThcnusamBfmeaHuRAulAhC488PlKhVbnAbi2pdxVnqqjyvajlQ==)
```

推送已经完成，并在 Azure 中构建了代码，现在您可以使用 URL 来检查您的应用程序是否正在运行。

```
(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ curl [https://simplefuncclass.azurewebsites.net/api/httpexample?code=c0U7ThcnusamBfmeaHuRAulAhC488PlKhVbnAbi2pdxVnqqjyvajlQ==](https://simplefuncclass.azurewebsites.net/api/httpexample?code=c0U7ThcnusamBfmeaHuRAulAhC488PlKhVbnAbi2pdxVnqqjyvajlQ==)
Please pass a name on the query string or in the request body(myvenv) jugs@jugs:~/Desktop/azurefxDemo$ curl [https://simplefuncclass.azurewebsites.net/api/httpexample?code=c0U7ThcnusamBfmeaHuRAulAhC488PlKhVbnAbi2pdxVnqqjyvajlQ==&name=Azure%20Function](https://simplefuncclass.azurewebsites.net/api/httpexample?code=c0U7ThcnusamBfmeaHuRAulAhC488PlKhVbnAbi2pdxVnqqjyvajlQ==&name=Azure%20Function)
Hello Azure Function!
```

现在，我们已经完成了最简单的应用程序。让我们试试您现有的应用程序。在我的例子中，我有这个文本分类，它作为一个 web 服务在一个实际的应用程序中运行。我将使用无服务器使它可用。

```
jugs@jugs:~/Desktop/LocalFunctionProj$ tree
.
├── host.json
├── HttpExample
│   ├── function.json
│   ├── host.json
│   ├── __init__.py
│   ├── prediction.py
│   ├── __pycache__
│   │   ├── __init__.cpython-36.pyc
│   │   ├── prediction.cpython-36.pyc
│   │   └── tokenization.cpython-36.pyc
│   ├── tokenization.py
│   └── vocab.txt
├── local.settings.json
└── requirements.txt2 directories, 12 files
```

这些是我创建的文件列表，我将展示文件的内容。首先，`__init__.py`文件看起来像这样。

```
import loggingimport azure.functions as func
from . import prediction as preddef main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')if name:
        res = pred.predict(name)
        return func.HttpResponse(f"Your query = {name} \n\nLabel = {res}!")
    else:
        return func.HttpResponse(
             "Please pass a name on the query string or in the request body",
             status_code=400
        )
```

我在这个主函数中导入了`prediction`类，以便能够使用它的方法。

prediction.py 文件如下所示。

```
import json
import requests
import numpy as np
from . import tokenization
import argparse
import os
import urllib.request
import pathlibdef get_query(query):
  return query + "now new query"def predict(query):
    query = query #request.args.get('text')endpoints = "[http://1x.1xx.1x3.1x8:85xx/v1/models/myclassifier:predict](http://18.162.113.148:8501/v1/models/newchicclassifier:predict)"
    headers = {"content-type": "application-json"}filepath = pathlib.Path(__file__).parent / 'vocab.txt'tokenizer = tokenization.FullTokenizer(vocab_file=filepath, do_lower_case=True)........ 
Follow the previous tutorial for details
........label_id = 0instances = [{"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_id}]data = json.dumps({"signature_name": "serving_default", "instances": instances})response = requests.post(endpoints, data=data, headers=headers, verify=False)
    prediction = json.loads(response.text)['predictions']
    idx_res = np.argmax(prediction)with urllib.request.urlopen("[https://myexample.s3-ap-southeast-1.amazonaws.com/catDict.txt](https://fashion-demo-assets.s3-ap-southeast-1.amazonaws.com/catDict.txt)") as f:
      data = f.readlines()catlib = {}
    for l in data:
      strsplit = (l.decode('utf-8')).split('\t')
      catlib[strsplit[1].strip()] = strsplit[0].strip()result = catlib[str(idx_res)]
return result
```

predict 方法中的变量名`endpoint`的值是我在 AWS 中运行的服务，你可以参考我以前关于提供 BERT 服务的博客来了解更多信息。`tokenization.py`和`vocab.txt`也来自伯特。此外，我将类别/标签列表作为 AWS s3 存储文件。

在`requirement.txt`文件中添加所有 python 依赖项。

```
azure-functions
tensorflow==1.14
Pillow
requests
....
```

一旦全部完成，我们就可以使用相同的 Azure Function 命令行将代码推送到 Azure。这个概念也融合了 CI/CD 的思想。推送的输出与前一个相同。我将使用推送产生的 URL，并在终端中卷曲或检查浏览器。这是输出。

![](img/8066453c698af713e03a99e61e51d4f0.png)

图:Azure 函数输出