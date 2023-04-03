# 如何用 Python 为 Kubernetes 编写验证和变更准入控制器 Webhooks

> 原文：<https://medium.com/analytics-vidhya/how-to-write-validating-and-mutating-admission-controller-webhooks-in-python-for-kubernetes-1e27862cb798?source=collection_archive---------1----------------------->

![](img/688c2ee376dee92147104fc99c34a4a1.png)

Mattia Serrani 在 [Unsplash](https://unsplash.com/search/photos/gate?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

准入控制器 Webhooks 是控制部署到 Kubernetes 集群的一个很好的方法。在这个故事中，我们将介绍如何使用 Python web framework Flask 来编写这两者。假设读者对 Kubernetes 中资源如何工作以及如何将 Flask 应用程序部署到 Kubernetes 集群有一些基本的了解。

## **准入控制员 Webhook 是做什么的？**

创建、修改或删除 Kubernetes 资源时，会触发准入控制器 Webhook。本质上，一个 HTTP 请求被发送到一个名称空间中指定的 Kubernetes 服务，该服务返回一个 JSON 响应。根据该响应，采取行动。

有两类准入控制器，验证和变异。验证接纳控制器验证传入的请求，并基于自定义逻辑返回二进制响应“是”或“否”。例如，如果 Pod 资源没有特定的标签，请求将被拒绝，并显示一条消息说明原因。变异准入控制器基于定制逻辑修改传入请求。一个例子是，如果入口资源没有正确的注释，将添加正确的注释，并且该资源将被接纳。

在上述场景中，准入控制器非常强大，可以非常精确地控制 Kubernetes 集群的进出。现在让我们开始吧。

*注意:下面显示的代码示例已经在 Kubernetes 版本* ***1.14.2 中测试过。*** *请阅读文档获取最新版本。*

**编写验证准入控制器 Webhook**

开始之前，让我们创建一个规范:

*   我们将验证一个部署资源，以检查它是否有标签。如果标签不存在，那么我们拒绝该请求。
*   当创建新部署时，会发生上述情况。

*validating _ admission _ controller . py*

```
from flask import Flask, request, jsonifyadmission_controller = Flask(__name__)[@admission_controller](http://twitter.com/admission_controller).route('/validate/deployments', methods=['POST'])
def deployment_webhook():
    request_info = request.get_json() if request_info["request"]["object"]["metadata"]["labels"].get("allow"):
        return admission_response(True, "Allow label exists") return admission_response(False, "Not allowed without allow label")def admission_response(allowed, message):
    return jsonify({"response": {"allowed": allowed, "status": {"message": message}}})if __name__ == '__main__':
    admission_controller.run(host='0.0.0.0', port=443, ssl_context=("/server.crt", "/server.key"))
```

在上面，我们编写了一个函数，它执行以下操作:

*   获取传入的请求对象。在我们的例子中，当我们注册下面的验证控制器时，我们可以假设该对象将始终是一个部署资源。
*   检查“允许”标签是否存在。返回一个 JSON HTTP 响应，如果存在的话，将允许的布尔值设置为 True，并显示一条消息。
*   如果不满足上述条件，我们将始终拒绝创建任何部署资源。

现在可以部署上述内容。我们不会深入讨论如何部署上述内容的细节，但我会快速列出要点:

*   从安装了 Flask 的 *admission_controller.py* 创建一个 Docker 映像。
*   生成自签名 CA，生成 csr 和证书，然后基于此证书创建一个秘密。
*   从名称空间中创建的 Docker 映像创建部署。该服务必须通过 SSL 进行保护。将上一步中创建的机密作为卷安装到部署中。
*   在与部署相同的名称空间中创建指向正确端口的服务。

要注册我们的验证控制器，我们需要应用以下配置:

*validating _ admission _ web hook . YAML*

```
apiVersion: admissionregistration.k8s.io/v1beta1
kind: ValidatingWebhookConfiguration
metadata:
  name: validating-webhook
  namespace: test
webhooks:
  - name: test.example.com
    failurePolicy: Fail
    clientConfig:
      service:
        name: test-validations
        namespace: test
        path: /validate/deployments
      caBundle: <redacted> # a base64 encoded self signed ca cert is needed because all Admission Webhooks need to be on SSL
    rules:
      - apiGroups: ["apps"]
        resources:
          - "deployments"
        apiVersions:
          - "*"
        operations:
          - CREATE
```

在上面的例子中，我们可以看到我们只对部署感兴趣，因为我们缩小了 web 钩子将执行的资源。CN(在我们的例子中)deployment-test.test.svc 需要一个自签名的 SSL 证书。

我们可以通过运行以下命令来应用上述内容:

```
kubectl apply -f validating_admission_webhook.yaml
```

要进行验证，请在元数据部分创建一个带有“allow”标签的新部署，可以使用任何值，这样就应该创建了。比如我们可以尝试部署 nginx。

```
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  labels:
    app: nginx
  name: nginx
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1
        ports:
        - name: http
          containerPort: 80
```

上述操作将失败，并显示以下消息:

```
Error from server: error when creating "nginx.yaml": admission webhook "test.example.com" denied the request: Not allowed without allow label
```

尝试添加允许标签，看看它是否通过。如果你需要帮助，请在评论区联系我们。

**编写一个变异准入控制器 Webhook**

开始之前，让我们创建一个规范:

*   我们将改变一个部署资源和一个标签。
*   当创建新部署时，会发生上述情况。

*变异 _ 准入 _ 控制器. py*

```
from flask import Flask, request, jsonify
import base64
import jsonpatchadmission_controller = Flask(__name__)[@admission_controller](http://twitter.com/admission_controller).route('/mutate/deployments', methods=['POST'])
def deployment_webhook_mutate():
    request_info = request.get_json()
    return admission_response_patch(True, "Adding allow label", json_patch = jsonpatch.JsonPatch([{"op": "add", "path": "/metadata/labels/allow", "value": "yes"}]))def admission_response_patch(allowed, message, json_patch):
    base64_patch = base64.b64encode(json_patch.to_string().encode("utf-8")).decode("utf-8")
    return jsonify({"response": {"allowed": allowed,
                                 "status": {"message": message},
                                 "patchType": "JSONPatch",
                                 "patch": base64_patch}})
if __name__ == '__main__':
    admission_controller.run(host='0.0.0.0', port=443, ssl_context=("/server.crt", "/server.key"))
```

在上面，我们编写了一个函数，它执行以下操作:

*   获取传入的请求对象。在我们的例子中，我们可以假设当我们注册下面的变异控制器时，该对象将始终是一个部署资源。
*   添加一个值为“yes”的 allow 标签，并在响应中返回 base64 编码的 JSONPatch。

部署细节类似于验证准入控制器，请参考上一节。要注册我们的变异控制器，我们需要创建以下内容。

*mutating _ admission _ web hook . YAML*

```
apiVersion: admissionregistration.k8s.io/v1beta1
kind: MutatingWebhookConfiguration
metadata:
  name: mutating-webhook
  namespace: test
  labels:
    component: mutating-controller
webhooks:
  - name: test.example.com
    failurePolicy: Fail
    clientConfig:
      service:
        name: test-mutations
        namespace: test
        path: /mutate/deployments
      caBundle: <redacted> # a base64 encoded self signed ca cert is needed because all Admission Webhooks need to be on SSL
    rules:
      - apiGroups: ["apps"]
        resources:
          - "deployments"
        apiVersions:
          - "*"
        operations:
          - CREATE
```

在上面的例子中，我们可以再次看到，我们只对部署感兴趣。CN(在我们的例子中)test-mutations.test.svc 需要一个自签名的 SSL 证书。

我们可以通过运行以下命令来应用上述内容:

```
kubectl apply -f mutating_admission_webhook.yaml
```

要进行验证，请创建一个新部署，允许标签将添加到新部署中:

```
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  ...
  labels:
    **allow: "yes"**
    app: nginx
  name: nginx
  ...
```

感谢您的阅读，我希望这是有用的。如果你遇到任何问题，请在评论区回复。