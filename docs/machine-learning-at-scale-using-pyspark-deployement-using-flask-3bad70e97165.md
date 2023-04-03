# 使用 Pyspark 进行大规模机器学习，使用 AzureML/Flask 进行部署

> 原文：<https://medium.com/analytics-vidhya/machine-learning-at-scale-using-pyspark-deployement-using-flask-3bad70e97165?source=collection_archive---------6----------------------->

![](img/87539fd841285f2fb9a2533f4fcfd2ea.png)

照片由[萨法尔·萨法罗夫](https://unsplash.com/@codestorm?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

大家好，在过去的几个月里，我一直在研究可扩展性&生产机器学习算法。我在网上搜索了很多，得到的支持很少。公司仍在努力在这一领域获得更高的成功率。根据调查，只有 4%的 ML 模型用于部署和生产环境，这是因为社区对此的支持较少。让我们从今天的话题开始，为社区做一点贡献。

# **在我们继续之前，需要了解一些事情:-**

1.  为什么选择 Pyspark 和 databricks 笔记本？
2.  什么是烧瓶，什么是替代品？

我用过 pyspark 和 Databricks notebook，因为它很好地定义了显示 spark 数据帧和图形的功能。Databricks 还提供集群设置，因此我们可以在集群中使用任何机器配置来获得更高的计算能力。在我们的例子中，我使用了(3 台机器 42GB 内存)。您可以查看【这里】([https://databricks.com/](https://databricks.com/))了解更多信息和使用方法。

这里使用 Flask 进行部署。Flask 是一个用于 python 的 web 服务器，它帮助创建公开服务请求的端点的服务器。这不是强制性的，我们必须去与烧瓶，有许多替代品目前在市场上。稍后我们将讨论更多关于这一点以及如何使用 etc..更多信息请查看(this)[[http://flask.pocoo.org/](http://flask.pocoo.org/)

我选择非常基本的数据集，泰坦尼克号数据集在这里使用，但任何一个可以玩其他数据集。这里执行的一些基本预处理操作，如特征提取、插补、删除等..我将以函数的方式展示每一步。

**步骤 1 :-导入所有重要的库**

```
*"""*
*Loading important package of spark* 
*"""*
**from** **pyspark.sql** **import** SparkSession
**from** **pyspark.ml** **import** Pipeline,PipelineModel
**from** **pyspark.sql.functions** **import** *
**from** **pyspark.ml.pipeline** **import** Transformer,Estimator
**from** **pyspark.ml.feature** **import** StringIndexer,VectorAssembler
**from** **pyspark.ml.classification** **import** LogisticRegression
**from** **pyspark.ml.tuning** **import** CrossValidator, ParamGridBuilder
```

这里我们用的是`pipeline`，基本上工作在类似顺序操作的阶段。在后面的代码中，你会看到我是如何为`StringIndexer`、`VectorAssembler`和`algorithm`使用管道的

其他进口库请在 spark 官网查询。

**步骤 2:创建 Spark 会话**

```
*"""*
*Spark session creater* 
*"""*

st = SparkSession \
        .builder \
        .appName('Titanic') \
        .getOrCreate()
```

您也可以为这个会话设置许多自定义内存选项，为了简单起见，我使用默认配置。

**步骤 3 在 Spark 数据帧中加载数据集**

```
*"""*
*Load data function for loading data..*
*@param -* 
 *path - path of file*
 *header_value - header value, incase true first row will be header*

*@return - dataframe of loaded intended data.*
*"""*

**def** load_data(path,header_value):
  df = st.read.csv(path,inferSchema=**True**,header=header_value)
  **return** dfdf = load_data('/FileStore/tables/titanic_train.csv',**True**) 
df_test = load_data('/FileStore/tables/titanic_test.csv',**True**)
```

加载数据文件，在这种情况下，训练和测试数据文件是分开的。为了方便起见，我创建了一个函数，这样每次想要加载数据时它都会调用函数`load_data`。

**步骤 4 为预处理数据创建一个定制的转换器**

```
*'''*
*Custom Transformer class for tranformation implementation .*

*@param -* 
 *Transformer - Transformer class refrence* 
 *df - dataframe in which operation need to be carried ( passed through tranform function)*
 *A - A class for variable sharing.*

*@return -*
 *df - a dataframe which contains prediction value as well with featured value.* 

*'''*

**class preprocess_transform**(Transformer):

    **def** _transform(self,df):
      print("********************************  in Transform method ...************************************")

      *"""*
 *Generate feature column in dataframe based on specific logic*

 *@param -* 
 *df - dataframe for operation.*

 *@return -* 
 *df - dataframe with generated feature.*
 *"""*

      **def** feature_generation(self,df):
        df = df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))
        df = df.replace(['Mlle','Mme', 'Ms', 'Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr',  'Mrs',  'Mrs',  'Other',  'Other','Other','Mr','Mr','Mr'])
        df = df.withColumn("Family_Size",col('SibSp')+col('Parch'))
        df = df.withColumn('Alone',lit(0))
        df = df.withColumn("Alone",when(df["Family_Size"] ==0, 1).otherwise(df["Alone"]))
        **return** df

      *"""*
 *Impute Age based on Age mean of specific gender. ex for male mean is 46 update all null male row with 46, similarly for others*

 *@param -* 
 *df - dataframe for operation*

 *@return -*
 *df - with imputed value*

 *"""*

      **def** Age_impute(self,df):
        Age_mean = df.groupBy("Initial").avg('Age')
        Age_mean = Age_mean.withColumnRenamed('avg(Age)','mean_age')
        Initials_list = Age_mean.select("Initial").rdd.flatMap(**lambda** x: x).collect()
        Mean_list = Age_mean.select("mean_age").rdd.flatMap(**lambda** x: x).collect()
        **for** i,j **in** zip(Initials_list,Mean_list):
            df = df.withColumn("Age",when((df["Initial"] == i) & (df["Age"].isNull()), j).otherwise(df["Age"]))

        **return** df

      *"""*
 *Impute Embark based on mode of embark column*
 *@param -* 
 *df - dataframe for operation*

 *@return -*
 *df - with imputed value*

 *"""*
      **def** Embark_impute(self,df):
        mode_value = df.groupBy('Embarked').count().sort(col('count').desc()).collect()[0][0]
        df = df.fillna({'Embarked':mode_value})
        **return** df

      *"""*
 *Impute Fare based on the class which he/she had sat ex: class 3rd has mean fare 9 and null fare belong to 3rd class so fill 9*
 *@param -* 
 *df - dataframe for operation*

 *@return -*
 *df - with imputed value*

 *"""*
      **def** Fare_impute(self,df):
        Select_pclass = df.filter(col('Fare').isNull()).select('Pclass')
        **if** Select_pclass.count() > 0:
          Pclass = Select_pclass.rdd.flatMap(**lambda** x: x).collect()
          **for** i **in** Pclass:
            mean_pclass_fare = df.groupBy('Pclass').mean().select('Pclass','avg(Fare)').filter(col('Pclass')== i).collect()[0][1]
            df = df.withColumn("Fare",when((col('Fare').isNull()) & (col('Pclass') == i),mean_pclass_fare).otherwise(col('Fare')))
        **return** df

      *'''*
 *combining all column imputation together..*

 *@param -* 
 *df - a dataframe for operation.*

 *@return -* 
 *df - dataframe with imputed value.*

 *'''*
      **def** all_impute_together(df):
        df = Age_impute(self,df)
        df = Embark_impute(self,df)
        df = Fare_impute(self,df)
        **return** df

      *'''*
 *converting string to numeric values.*

 *@param -* 
 *df - dataframe contained all columns.*
 *col_list - list of column need to be* 

 *@return -* 
 *df - transformed dataframe.*
 *'''*
      **def** stringToNumeric_conv(df,col_list):
        indexer = [StringIndexer(inputCol=column,outputCol=column+"_index").fit(df) **for** column **in** col_list]
        string_change_pipeline = Pipeline(stages=indexer)
        df = string_change_pipeline.fit(df).transform(df)
        **return** df

      *"""*
 *Drop column from dataframe*
 *@param -*
 *df - dataframe* 
 *col_name - name of column which need to be dropped.*
 *@return -*
 *df - a dataframe except dropped column*
 *"""*
      **def** drop_column(df,col_list):
        **for** i **in** col_list:
            df = df.drop(col(i))
        **return** df

      col_list = ["Sex","Embarked","Initial"]
      dataset = feature_generation(self,df)
      df_impute = all_impute_together(dataset)
      df_numeric = stringToNumeric_conv(df_impute,col_list)
      df_final = drop_column(df_numeric,['Cabin','Name','Ticket','Family_Size','SibSp','Parch','Sex','Embarked','Initial'])
      **return** df_final
```

在变压器类中存在各种方法，每种方法用于不同的操作。

***Feature _ generation()-从名称列生成标题。***

***Age_impute() —根据年龄组平均值估算年龄*** 。

***登船 _ 估算&费用 _ 估算—估算登船和费用***

***StringToNumeric()-字符串数据类型为数值***

***Drop_col —从数据帧中删除不需要的列***

**步骤 5 创建管道并提取模型**

```
**from** **pyspark.ml.classification** **import** GBTClassifier 
**from** **pyspark.ml.classification** **import** RandomForestClassifier 
**from** **pyspark.ml.evaluation** **import** MulticlassClassificationEvaluator *# initialization for pipeline setup* 
my_model = preprocess_transform() 
df = my_model.transform(df) feature = VectorAssembler(inputCols=['Pclass','Age','Fare','Alone','Sex_index','Embarked_index','Initial_index'],outputCol="features") rf = RandomForestClassifier(labelCol="Survived", featuresCol="features", numTrees=10) *'''* *pipeline stages initilization , fit and transform.* *'''* 
pipeline = Pipeline(stages=[feature,rf]) *model = pipeline.fit(df)*  
paramGrid = ParamGridBuilder().addGrid(rf.numTrees,[100,300]).build() evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy") crossval = CrossValidator(estimator=pipeline,                           estimatorParamMaps=paramGrid,                        evaluator=evaluator,numFolds=3)  
*# use 3+ folds in practice*  *# Run cross-validation, and choose the best set of parameters.* 
cvModel = crossval.fit(df) 
prediction = cvModel.transform(*df_test*)   
*mlflow.spark.log_model(model, "spark-model16")  mlflow.spark.save_model(model, "spark-model_test")* 
```

在这些步骤中，首先我们调用 transformer 类，并用我们创建的上述选项转换我们的数据帧。

变换后调用 FeatureAssembler，它用于将所有输入特征绑定到一个矢量中。创建一个管道对象要经过两个阶段，第一个是特征组装器，第二个是估计器(分类器或回归器)。

如果你愿意，你可以创建一个交叉验证使用的超参数列表，paramGridBuilder 使用列表值来分配各种超参数。

为测量标准设置评估器，如多类分类评估器

调用 crossvalidator 并传递管道模型、参数网格、赋值器和折叠数。

一旦这些都完成了，将数据放入 crossvalidator 函数，记住它包含一个有模型的管道。所以它会用不同的参数组合训练模型，最后用`transform`的方法得到预测数据。

[MLFlow](https://mlflow.org/) 是一个管理机器学习周期的平台。在预测之后，我们可以使用 ml 流的两个函数，即`log`和`save`。日志功能将在 ml 流门户中记录处理指标，保存功能将保存最佳 ML 模型。ML flow 还有许多其他有用的功能，所以只需查看它们的官方文档。

**使用 Azure ML 在 Azure 中进行第 6 步部署**

```
**import** mlflow.azureml
**from** azureml.core **import** Workspace
**from** azureml.core.webservice **import** AciWebservice, Webservice
```

在 Azure ML 中创建一个工作区。

```
workspace_name = "MLServiceDMWS11"
subscription_id = "xxxxxxxx-23ad-4272-xxxx-0d504b07d497"
resource_group = "mlservice_ws"
location = "xxx"
azure_workspace = Workspace.create(name=workspace_name,
                                   subscription_id=subscription_id,
                                   resource_group=resource_group,
                                   location=location,
                                   create_resource_group=**False**,
                                  )
```

在工作空间中建立模型的图像，这基本上意味着我们只是在工作空间中保存模型对象。

```
azure_image, azure_model = mlflow.azureml.build_image(model_uri="/dbfs/databricks/mlflow/my_test_ml_flow",
                                                      workspace=azure_workspace,
                                                      description="model_description",
                                                      synchronous=**True**)
```

使用 Azure web 服务 api 会将模型公开为 rest 端点。传递模型图像、工作空间和配置设置。

```
webservice_deployment_config = AciWebservice.deploy_configuration()
webservice = Webservice.deploy_from_image(deployment_config=webservice_deployment_config,
                                          image=azure_image, 
                                          workspace=azure_workspace, 
                                          name='mysvc')
webservice.wait_for_deployment()
print("Scoring URI is: %s", webservice.scoring_uri)
```

一旦部署了模型，让我们检查一下我们创建的 API 是否工作。在列表中传递参数和相应的值，并点击 post 请求。一旦请求成功，它将通过响应进行确认。采用 i/p 和 o/p 的标准方式是 json 格式。

```
**import** requests
**import** json

sample_input = {
    "columns": [
        "col1",
        "col2",
        "col3",
       "coln "
    ],
    "data": [
        [val1, val2, val3,...... valn]
    ]
}
response = requests.post(
              url=webservice.scoring_uri, data=json.dumps(sample_input),
              headers={"Content-type": "application/json"})
response_json = json.loads(response.text)
print(response_json)
```

这只是这篇文章的初稿，我会在不久的将来更新。你可以在下面我的 git 链接中查看全部代码。如果您有任何反馈或建议，我们将不胜感激。

 [## yug 95/机器学习

### 使用 PySpark Titanic Survival 分类器使用 flask web app 部署，并公开一个 rest 端点。文件信息…

github.co](https://github.com/yug95/MachineLearning/tree/master/flask_app_deployment) 

感谢您的支持！我们下次再见:)