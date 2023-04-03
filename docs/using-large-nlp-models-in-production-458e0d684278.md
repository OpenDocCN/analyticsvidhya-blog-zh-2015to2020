# 在生产中使用大型 NLP 模型

> 原文：<https://medium.com/analytics-vidhya/using-large-nlp-models-in-production-458e0d684278?source=collection_archive---------13----------------------->

![](img/e39a3b9b084d578dbc45c9784a1d2ca3.png)

来源:https://www.reshot.com/

来自变压器的双向编码器表示(BERT)是 Google 开发的基于变压器的机器学习技术，用于自然语言处理(NLP)预训练[1]。在大型数据集上训练的 BERT 模型或其他 NLP 模型利用更多的内存。例如，BERT-base 需要大约 450 MB 的内存，而 BERT-large 需要大约 1.2 GB 的内存。

当使用这些 NLP 模型中的任何一个来微调我们的数据集时，我们保存多个版本的微调数据集。在生产中，我们可能需要在运行时加载任何版本。作为一种惯例，我们可以考虑在每次发出 web 请求时加载 NLP 模型，然而这在低计算生产环境中可能不可行。将一个 BERT 模型从 model mongoDB 加载到本地，然后再将本地模型加载到内存中是非常耗时的。所以在这篇文章中，我想描述一下我在生产中处理大型 NLP 模型的方法。

出于演示目的，我将使用一个简单的 Scikit-learn 模型。

**属地**

```
scikit-learnjoblib
PyMongo
```

让我们创建一个简单的 sklearn 模型[2]:

```
import joblib
from sklearn import svm
from sklearn import datasetsclf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)model_name = 'mymodel_v1'
model_fpath = f'{model_name}.joblib'
joblib.dump(clf, model_fpath)
```

现在模型保存在本地，现在我们需要将它保存到 DB，以便我们可以在生产中从 DB 加载。出于演示目的，我将使用 MongoDB。

```
# internal
import datetime
# external
import gridfs
import pymongo# create mongo client to communicate with mongoDB
mc = pymongo.MongoClient(host='220.24.52.190',
                         port=27017)
# load or create database
mydb = mc.test_database
# load / create file system collection
fs = gridfs.GridFS(mydb)
# load / create model status collection
mycol = mydb['ModelStatus']
# save the local file to mongodb
with open(model_fpath, 'rb') as infile:
    file_id = fs.put(
        infile.read(), 
        model_name=model_name
    )
# insert the model status info to ModelStatus collection 
params = {
    'model_name': model_name,
    'file_id': file_id,
    'inserted_time': datetime.datetime.now()
}
result = mycol.insert_one(params)
```

现在模型保存在 mongoDB 中，可以在生产过程中检索。当从数据库中检索模型时，我们将使用元类遵循单例设计模式。下面是基类的代码:

```
class ModelSingleton(type):
    """
    Metaclass that creates a Singleton base type when called.
    """
    _mongo_id = {} def __call__(cls, *args, **kwargs):
        mongo_id = kwargs.pop('mongo_id')
        if mongo_id not in cls._mongo_id:
            print('Adding model into ModelSingleton')
            cls._mongo_id[mongo_id] = super(ModelSingleton, cls)\
                .__call__(*args, **kwargs)
        return cls._mongo_id[mongo_id]
```

加载模型的代码如下:

```
class LoadModel(metaclass=ModelSingleton):
    def __init__(self, *args, **kwargs):
        self.mongo_id = kwargs['mongo_id']
        self.clf = self.load_model()

    def load_model(self):
        print('loading model')
        f = fs.find({"_id": ObjectId(self.mongo_id)}).next()
        with open(f'{f.model_name}.joblib', 'wb') as outfile:
            outfile.write(f.read())
        return joblib.load(f'{f.model_name}.joblib')
```

现在我们只需要检查 mongo **_id** 在模型版本中的任何变化。仅从数据库获取 **_id** 的代码如下:

```
result = mycol.find({"filename": model_name}, {'_id': 1})\
    .sort('uploadDate', -1)
if result.count():
    mongo_id = str(result[0]['_id'])
```

将模型加载到生产环境中的代码如下:

```
model = LoadModel(mongo_id=mongo_id)
clf = model.clf
```

现在，只有当数据库中发生变化时，才会从数据库中下载模型，否则将从内存中取出模型。

编码快乐！！！

参考资料:

[1][https://en . Wikipedia . org/wiki/BERT _(language _ model)](https://en.wikipedia.org/wiki/BERT_(language_model))

[2][https://sci kit-learn . org/stable/modules/model _ persistence . html](https://scikit-learn.org/stable/modules/model_persistence.html)