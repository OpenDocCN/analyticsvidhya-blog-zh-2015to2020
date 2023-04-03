# MongoDB 漫游

> 原文：<https://medium.com/analytics-vidhya/a-walkthrough-of-mongodb-aef402594144?source=collection_archive---------11----------------------->

![](img/48a5c9f1f699bdcfffae5332c307a669.png)

# 什么是 MongoDB？

MongoDB 是一个 NoSQL 数据库——NoSQL 数据库是一个不用 SQL 查询的数据库。除此之外，NoSQL 对于定义数据库来说没有任何意义。所以让我们再来定义一下 MongoDB。MongoDB 是一个 JSON 文档数据库。它允许您存储和查询 JSON 风格的文档，并带有一些智能。这意味着您可以将包含嵌套数据的对象存储在一个集合中(集合就像 MongoDB 的表)。
对于事务型存储，它通常比传统的 SQL 数据库更快，但缺乏用于分析用途的查询能力。

![](img/9ed7116c27b30bda76d38e9f20ec23b1.png)

# 为什么是 MongoDB？

当我们需要一个 NoSQL 数据库时，我们选择了 MongoDB，原因有很多:

*   在云中运行良好(规模、容错、安全)
*   易于入门(大量清晰的文档)
*   全球许多大公司都在使用它，所以 MongoDB 有钱继续开发。

![](img/e973d2096b9d5fb8c4886a5a9194adb9.png)

# MongoDB 相对于 RDBMS 的优势:

*   MongoDB 是无模式的。它是一个文档数据库，其中一个集合包含不同的文档。但是在 RDBMS 中，你需要首先设计你的表，数据结构，关系，只有这样你才能开始编码。
*   字段数量、内容和文档大小可能会有所不同。但是在 RDBMS 中，每条记录都必须遵循特定的预定义模式。
*   MongoDB 是水平可扩展的，即我们可以添加更多的服务器(分片),但 RDBMS 只能垂直扩展，即增加 RAM。
*   MongoDB 强调 CAP 定理(一致性、可用性和分区容差)，而 RDBMS 强调 ACID 属性(原子性、一致性、隔离性和持久性)。
*   MongoDB 最适合分层数据存储，而 RDBMS 不适合。
*   MongoDB 支持 JSON 查询语言和 SQL，但是 RDBMS 只支持 SQL 查询语言。
*   与 RDBMS 相比，MongoDB 易于设置、配置和运行。它的 Java 客户端也很好用。
*   MongoDB 比 RDBMS 等传统数据库系统快近 100 倍，而 RDBMS 比 NoSQL 数据库慢。
*   MongoDB 中不支持复杂连接，但 RDBMS 支持复杂连接，这可能很难理解，并且执行起来需要太多时间。
*   MongoDB 使用内存来存储工作集，从而加快了访问速度。
*   MongoDB 支持深度查询能力，即我们可以使用基于文档的查询语言在文档上执行动态查询，这种语言几乎和 SQL 一样强大。
*   在 MongoDB 中，不需要将应用程序对象转换/映射到数据库对象。

让我们开始使用 python 滚动 MongoDB。

首先，安装 [pymongo](https://pypi.org/project/pymongo/) 包，使用 Python 实现 MongoDB 连接。

```
!pip install pymongo
```

![](img/7fd5a5af8960b3d26e5edda6ab237a08.png)

安装控制台

**积垢运行**

![](img/4c76aa5a5350fb5fc1f2ce3ae042b40a.png)

**导入包**

```
**import** pymongo
```

**设置数据库连接**

```
DEFAULT_CONNECTION_URL = "mongodb://localhost:27017/"
DB_NAME = "bank"# Establish a connection with mongoDB
client = pymongo.MongoClient(DEFAULT_CONNECTION_URL)# Create a DB
dataBase = client[DB_NAME]
```

MongoDB 安装在我的本地机器上。因此，作为本地主机提供的连接 URL。执行此命令后，数据库就创建好了。

**收藏**

MongoDB 将文档存储在集合中。集合类似于关系数据库中的表。

![](img/34747eb64de9d7d4daa32442adbfa57a.png)

```
UserDetails = dataBase['UserDetails']
```

上面的代码用于创建名为“UserDetails”的集合。

**插入**

```
record={'_id':1,
        'Name':'Antony',
        'EmployeeType':'Business',
        'Address':'VGN Cross street, Chennai',
        'AccountType':'Savings',
        'Age':45
       }
userDetails.insert_one(record)
```

验证 MongoDB Compass 中的数据，数据库“bank”已创建。

![](img/508338d4d9e970d9cb0055c6141d22f0.png)

数据库屏幕

验证该集合是以“UserDetails”的名称创建的

![](img/ca9bdcd07c73b4ace1f6ae5759ae10b0.png)

用户详细信息

一次插入多条记录。这可以通过下面的步骤很容易地实现，

```
records = [{'_id':2,       
        'Name':'Anto',
        'EmployeeType':'Business',
        'Address':'VGN Cross street, Chennai',
        'AccountType':'Savings',
        'Age':45
       },
{'_id':3,       
        'Name':'Chris',
        'EmployeeType':'**Self**',
        'Address':'Mayur street, Chennai',
        'AccountType':'Savings',
        'Age':31
       },
{'_id':4,
        'Name':'Maddy',
        'EmployeeType':'Business',
        'Address':'KVN street, Chennai',
        'AccountType':'Savings',
        'Age':56
       }
          ]
userDetails.insert_many(records)
```

![](img/c5b773bc7575eb1c2dcfc37e85f2ca84.png)

插入的记录

**显示**

***显示第一条记录:***

显示集合中的第一条记录

```
print(userDetails.find_one()){'_id': 1, 'Name': 'Antony', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}
```

***读取所有记录:***

```
**def** **display**():
    allRecords = userDetails.find()
    **for** idx, record **in** enumerate(allRecords):
        print(f"{idx}:{record}")**0**:{'_id': 1, 'Name': 'Antony', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}**1**:{'_id': 2, 'Name': 'Anto', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}**2**:{'_id': 3, 'Name': 'Chris', 'EmployeeType': 'Self', 'Address': 'Mayur street, Chennai', 'AccountType': 'Savings', 'Age': 31}**3**:{'_id': 4, 'Name': 'Maddy', 'EmployeeType': 'Business', 'Address': 'KVN street, Chennai', 'AccountType': 'Savings', 'Age': 56}
```

***显示有限的栏目:***

```
showLimitedColumns = userDetails.find({},{'Name'})
**for** idx, record **in** enumerate(showLimitedColumns):
    print(f"{idx}:{record}")**0**:{'_id': 1, 'Name': 'Antony'}
**1**:{'_id': 2, 'Name': 'Anto'}
**2**:{'_id': 3, 'Name': 'Chris'}
**3**:{'_id': 4, 'Name': 'Maddy'}
```

***应用查询或过滤:***

```
query={"Name" : "Antony"}
filterData = userDetails.find(query)
**for** **data** **in** filterData:
    print(**data**){'_id': 1, 'Name': 'Antony', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}
```

***使用条件运算符过滤数据:***

$ gt 用符号表示“大于”运算符。从下面的代码中选择年龄大于 45 岁的用户。

```
criteria ={"Age" :{"$gt" : 45}}
results = userDetails.find(criteria)
**for** rec **in** results: 
    print(rec){'_id': 4, 'Name': 'Maddy', 'EmployeeType':'Business', 'Address': 'KVN street, Chennai', 'AccountType': 'Savings', 'Age': 56}
```

**更新**

此处将更新用户 id“4 ”,名称为“Mark ”,而不是“Maddy”。基于筛选器更新集合中的单个文档。

```
presentData={"_id":4}
updateData={"$set": {"Name":"Mark"}}
userDetails.update_one(presentData, updateData)
```

核实数据

```
**display**()**0**:{'_id': 1, 'Name': 'Antony', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}
**1**:{'_id': 2, 'Name': 'Anto', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}
**2**:{'_id': 3, 'Name': 'Chris', 'EmployeeType': 'Self', 'Address': 'Mayur street, Chennai', 'AccountType': 'Savings', 'Age': 31}
**3**:{'_id': 4, 'Name': '**Mark**', 'EmployeeType': 'Business', 'Address': 'KVN street, Chennai', 'AccountType': 'Savings', 'Age': 56}
```

**删除**

在此删除名为“Chris”的第一条记录。从集合中移除单个文档。

```
query_to_Delete = {"Name":"Chris"}
userDetails.delete_one(query_to_Delete)
```

核实数据

```
**display**()**0**:{'_id': 1, 'Name': 'Antony', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}
**1**:{'_id': 2, 'Name': 'Anto', 'EmployeeType': 'Business', 'Address': 'VGN Cross street, Chennai', 'AccountType': 'Savings', 'Age': 45}
**2**:{'_id': 4, 'Name': 'Mark', 'EmployeeType': 'Business', 'Address': 'KVN street, Chennai', 'AccountType': 'Savings', 'Age': 56}
```

**删除&中的其他操作更新:**

***db . collection . delete many()***

从集合中删除所有匹配`filter`的文档。以下操作删除所有文件，其中`client : "Crude Traders Inc."`

```
**try** {
   db.orders.deleteMany( { "client" : "Crude Traders Inc." } );
} **catch** (e) {
   print (e);
}
```

***db . collection . update many()***

更新集合中与指定筛选器匹配的所有文档。

```
{ "_id" : 1, "name" : "Central Perk Cafe", "violations" : 3 }
{ "_id" : 2, "name" : "Rock A Feller Bar and Grill", "violations" : 2 }
{ "_id" : 3, "name" : "Empire State Sub", "violations" : 5 }
{ "_id" : 4, "name" : "Pizza Rat's Pizzaria", "violations" : 8 }
```

以下操作更新所有`violations`大于`4`且`[$set](https://docs.mongodb.com/manual/reference/operator/update/set/#up._S_set)`为待审核标志的文档:

```
**try** {
   db.restaurant.updateMany(
      { violations: { $gt: 4 } },
      { $set: { "Review" : **true** } }
   );
} **catch** (e) {
   **print**(e);
}
```

**MongoDB 图集**

**MongoDB Atlas** 是一个完全托管的云数据库，由构建 **MongoDB** 的同一批人开发。 **Atlas** 在您选择的云服务提供商(AWS、Azure 和 GCP)上处理部署、管理和修复部署的所有复杂性。

让我们开始滚动使用 Atlas 创建数据库和收集。

```
# Create a database connection
client = pymongo.MongoClient("mongodb+srv://anto:antouser@antocluster.rp5yu.mongodb.net/school?retryWrites=true")
db = client.school# create the student collection
student=db['student']# Insert new record
record={'_id':1, 'Name':'Ashok', 'standard':'V', 'Attendance %':78}
student.insert_one(record)
```

![](img/ae420548f5f28f99f10d07ffbc2888c2.png)

Atlas 中的样本数据库和集合

**图表**

图表如何在 MongoDB 中发挥作用？

这将是你要问的第一个问题。是的，我们可以使用 Atlas 通过仪表板管理来制作图表。默认情况下，Atlas 中的任何帐户都嵌入了电影数据库，以便制作图表。让我们看看它是如何显示的。

转到名为“图表”的选项卡

![](img/4c30d75a05dc2bb3389d004f2559c007.png)

仪表盘

单击示例仪表板上的。数据分别以线形图、条形图、地理空间分布图、饼图的形式呈现。

![](img/afa2a11c890d40c82cc0c7185332418f.png)

屏幕-I

![](img/a6b406ba3879356d8c23d14eb7d92975.png)

屏幕-II

我们也可以创建自己的定制仪表板。通过选择在 Atlas 帐户中创建的数据源。

![](img/7c925144463dce413fe89ad777eb0ded.png)

自定义仪表板

此外，我们可以向外部用户共享仪表板。

![](img/559a96d2851a289009bc1b5413713d48.png)

仪表板共享

我希望我的博客能让你对 MongoDB 有一个初步的了解。帮助您理解在 MongoDB 中从哪里开始。

快乐编码:)