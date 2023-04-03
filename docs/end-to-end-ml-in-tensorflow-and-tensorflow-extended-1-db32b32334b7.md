# Tensorflow 和 tensor flow Extended-1 中的端到端 ML

> 原文：<https://medium.com/analytics-vidhya/end-to-end-ml-in-tensorflow-and-tensorflow-extended-1-db32b32334b7?source=collection_archive---------10----------------------->

![](img/b4e035be0cc6bc30a5316bea0cd5a993.png)

萨法尔·萨法罗夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

一段时间以来，仍然有很多人问如何从 Tensorflow 或 Keras 部署训练好的模型，如何将 Keras 模型(h5)或检查点(ckpt)转换为冻结模型/原始缓冲区模型(pb)，或者如何将预训练模型转换为冻结模型。在所有这些之后，另一个问题将是我如何服务于这些模型，以及我如何从客户端进行推理。

有几种方法可以完成这些任务。服务可以使用基于 Python 的简单 web 框架 Flask 来完成，但是 Flask 有多可靠，不确定，有没有更好的方法来服务模型？是的，tensor flow-发球是关键。所有这些问题的答案就在这里，只需要很少几行代码和一个非常基本的模型。

因此，我将遵循这个管道，创建一个非常基本的网络，导出 TensorFlow 模型，使用 TensorFlow-serving 提供服务，并创建一个客户端来进行推理。

让我们创建一个只有一个输入节点和一个输出节点的非常基本的网络，我们将在本系列的下一部分讨论复杂架构。

```
import tensorflow as tfinput_placeholder = tf.placeholder(tf.int32, shape=[None, 1], name=’input_placeholder’)
output_placeholder = tf.identity(tf.add(input_placeholder, 2), name=’output_placeholder’)
```

因此，首先，我们导入张量流并创建一个名为 input_placeholder 的双节点 shape [None，1]，数据类型为 integer32，input_placeholder 的标识为 output_placeholder，并添加了标量。

现在，我们将模型作为 saved_model 导出到导出目录，如下所示。

```
import osexport_dir = “/tmp/tfmodeldir”
version = "1"
export_path = os.path.join(export_dir, version)sess = tf.Session()
tf.saved_model.simple_save(sess, export_dir, inputs={"key":input_placeholder}, outputs={"keys":output_placeholder})
```

上面几行将以 pb 格式将模型保存在特定的目录中。您可以在特定位置检查模型。您可以从终端使用 Tensorflow 的命令行 saved_model_cli 查看模型的签名和期望版本。

```
saved_model_cli show — dir /tmp/tfmodeldir/1 — allMetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['key'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 1)
        name: input_placeholder:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['keys'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 1)
        name: output_placeholder:0
  Method name is: tensorflow/serving/predict
```

到目前为止，我们都做得很好。因此，让我们从命令行使用 TensorFlow 服务于保存的模型。

```
tensorflow_model_server --port=850 --rest_api_port=8501 --model_name=simple_add --model_base_path=/tmp/tfmodeldir/2019-10-23 01:32:25.761732: I tensorflow_serving/model_servers/server.cc:82] Building single TensorFlow model file config:  model_name: simple_add model_base_path: /tmp/tfmodeldir/
2019-10-23 01:32:25.761842: I tensorflow_serving/model_servers/server_core.cc:462] Adding/updating models.
2019-10-23 01:32:25.761853: I tensorflow_serving/model_servers/server_core.cc:561]  (Re-)adding model: simple_add
2019-10-23 01:32:25.862420: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: simple_add version: 1}
2019-10-23 01:32:25.862486: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: simple_add version: 1}
2019-10-23 01:32:25.862515: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: simple_add version: 1}
2019-10-23 01:32:25.862566: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:363] Attempting to load native SavedModelBundle in bundle-shim from: /tmp/tfmodeldir/1
2019-10-23 01:32:25.862592: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /tmp/tfmodeldir/1
2019-10-23 01:32:25.862792: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2019-10-23 01:32:25.863032: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-10-23 01:32:25.892224: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:202] Restoring SavedModel bundle.
2019-10-23 01:32:25.892303: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:212] The specified SavedModel has no variables; no checkpoints were restored. File does not exist: /tmp/tfmodeldir/1/variables/variables.index
2019-10-23 01:32:25.892354: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:311] SavedModel load for tags { serve }; Status: success. Took 29757 microseconds.
2019-10-23 01:32:25.892390: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:103] No warmup data file found at /tmp/tfmodeldir/1/assets.extra/tf_serving_warmup_requests
2019-10-23 01:32:25.892505: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: simple_add version: 1}
2019-10-23 01:32:25.898323: I tensorflow_serving/model_servers/server.cc:324] Running gRPC ModelServer at 0.0.0.0:8500 ...
[evhttp_server.cc : 239] RAW: Entering the event loop ...
2019-10-23 01:32:25.900349: I tensorflow_serving/model_servers/server.cc:344] Exporting HTTP/REST API at:localhost:8501 ...
```

我们可以使用— tensorflow_model_server —帮助来检查参数定义，帮助将列出 tensorflow 服务中的几个参数。

我们程序中唯一剩下的部分是客户端部分。因为模型是用 rest API 和 grpc API 公开的。我们可以从客户端调用 REST 方法(具体来说是 POST)。

```
import requests
import jsonendpoint = “[http://localhost:8501/v1/models/simple_add:predict](http://localhost:8501/v1/models/simple_add:predict)"instance = [[11,1],[22,1],[44,1],[55,1]]headers = {“content-type”:”application-json”}
data = json.dumps({“signature-name”:”serving_default”, “instances”: instance})
response = requests.post(endpoint, data=data, headers=headers)
prediction = json.loads(response.text)[‘predictions’]
prediction
```

服务完模型后，HTTP 的端点会是这样的格式[HTTP://hostname:port/v1/models/model _ name:predict。](http://hostname:port/v1/models/model_name:predict.)