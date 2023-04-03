# 如何在 AWS 上部署 3 节点 MongoDB Shard 集群

> 原文：<https://medium.com/analytics-vidhya/how-to-deploy-3-node-mongodb-shard-cluster-on-aws-2bf2fd821e5?source=collection_archive---------2----------------------->

![](img/289be74bf48a07675e85947d67637d36.png)

萨法尔·萨法罗夫在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在本教程中，我们将在 AWS 中部署一个 MongoDB 分片集群。您可以尝试使用本教程将分片集群部署到任何云中，只需稍作修改。

目前，根据 MongoDB 指南，您需要 3 个节点作为配置服务器副本集，至少 3 个分片节点，其中每个节点是 3 个成员副本集和一个或多个 mongos 服务器。

根据官方文件，我们需要以下基础设施:

1.  配置服务器的 3 个 AWS EC2 实例(3 节点副本集)
2.  9 个 AWS EC2 实例，用于部署 3 个分片，其中每个分片都有一个由 3 个节点组成的副本集
3.  mongos 的 1 个 AWS EC2 实例

但是对于本教程，我们将使用下面的基础设施部署 shard 集群:

1.  配置服务器的 1 个 AWS EC2 实例(1 个节点副本集)
2.  2 个 AWS EC2 实例，用于部署 2 个分片，其中每个分片都有一个节点的副本集
3.  mongos 的 1 个 AWS EC2 实例

那么让我们开始吧—

# 配置服务器

在分片集群中，每个操作/查询都有特定的元数据存储在其中，这些元数据决定了数据存储的路径以及 MongoDB 的读写操作。

配置服务器是一个关键的数据库，它保存着操作分片集群的元数据信息

**部署:——**配置服务器需要部署在 3 节点副本集中，并且必须使其成为副本集。在本教程中，我们将部署一个单节点副本集。

# 蒙哥斯

Mongos 是一个作为主要主机工作的路由器，您可以使用 MongoDB 客户端库在具有任意数量 Mongos 主机的分片集群上运行。

**部署:——**配置服务器可以连接多个 mongos 路由器。因为他们没有自己的数据库。它们与配置服务器通信以操作分片集群。

拥有多台 mongos 服务器有助于在一台 mongos 服务器停机的情况下，客户端将使用另一台 mongos 服务器来查询集群。所有 mongodb 客户端库都支持多个 mongos 主机。

# 陶瓷或玻璃碎片

碎片是包含数据库和收集数据的 mongod 实例。数据的分发由配置服务器根据 mongos 服务器配置的分片密钥来决定。

**部署:——**根据 MongoDB 的指导方针，我们需要每个分片有一个 3 节点副本集。为简单起见，我们将在这里创建两个带有 1 个节点副本集的碎片。

你可以选择任何你想要的云(AWS/Oracle/AZure 等)。对于本文，我们将使用 AWS。

# 创建 3 个 Ubuntu 实例并命名为

1.  配置服务器/mongos 的 1 个服务器实例(config_instance)
2.  2 台服务器用于分片(分片 1 _ 实例/分片 2 _ 实例)

# MongoDB 安装:

可以遵循 MongoDB 指南—[https://docs . MongoDB . com/manual/tutorial/install-MongoDB-on-Ubuntu/](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/)

**MongoDB 版本:**

如果您试图将任何类型的数据恢复到分片集群中，您必须小心谨慎。由于事务支持，mongorestore 在 4.2.x 以上无法工作

```
** Notes on MongoDB site - [mongodump](https://docs.mongodb.com/manual/reference/program/mongodump/#bin.mongodump) and [mongorestore](https://docs.mongodb.com/manual/reference/program/mongorestore/#bin.mongorestore) **cannot** be part of a backup strategy for 4.2+ sharded clusters that have sharded transactions in progress, as backups created with [mongodump](https://docs.mongodb.com/manual/reference/program/mongodump/#bin.mongodump) *do not maintain* the atomicity guarantees of transactions across shards.
```

# 安装程序

1.  Ssh 到 config_instance 并从给定的链接安装 MongoDB。MongoDB 将在/etc/mongod.conf 位置放置一个配置文件
2.  打开/etc/mongod.conf 文件，通过更改属性 dbPath 和 systemLog path 来更改数据目录或日志目录。如果您正在更改默认目录，不要忘记使用下面的命令将所有权授予在 mongodb 安装期间由创建的 mongodb 用户

`sudo chown mongodb:mongodb /volume_paths -R`

3.安装后，使用命令运行 mongod 服务

`sudo service mongod restart`

4.如果您想继续使用密钥文件安全性，让我们也创建一个密钥文件

```
openssl rand - base64 756 > /home/ < userename > /
chmod 400 shardauthkey
```

5.现在在你选择的 vim/nano 等文本编辑器中打开/etc/mongod.conf。并将下面的设置复制到/etc/mongod.conf 文件的底部

```
sharding:
 clusterRole: configsvr
replication:
 replSetName: config
net:
 bindIp: 0.0.0.0
```

6.我们已经声明了服务器集群角色，并在 mongodb 配置中设置了副本集名称。

7.让我们重新启动 mongod 服务。重启后，让我们进入 mongod shell

8.使用下面的命令进入 mongod shell。这里的端口是 27017，这是在 mongod.conf 文件中配置的默认端口

`mongod --port=27017`

9.mongod shell 打开后，输入以下命令

```
{
    _id: "config",
    configsvr: true,
    members: [{
        _id: 0,
        host: "<interanl/elastic_ip_of_config_server>:27017"
    }, ]
}
```

您将看到一个副本集已经启动。我们现在已经启动了一个单节点副本集。如果需要，可以在副本集中添加更多实例。过一会儿你会看到外壳变成了初级。

10.现在设置一个默认用户进行认证，这是非常重要的。打开 mongo shell 后输入命令

```
use admin
db.createUser({
    user: "config",
    pwd: "config@123",
    roles: [{
        role: "root",
        db: "admin"
    }]
})
```

11.退出终端，现在使用下面的命令再次进入 mongo shell:

`mongo --port=2707 -u config -p config@123 --authenticationDatabase=admin`

12.再次打开配置文件/etc/mongod.conf，并在文件底部添加以下行

```
security:
 keyfile: /home/<username>/shardauthkey
```

重新启动 mongod 服务。我们已经设置了我们的配置服务器。

# 分片服务器:

**碎片-1:**

1.  Ssh 到我们在本教程开始时创建的 shard1_instance 和 shard2_instance。您可以用下面提到的相同步骤设置两个碎片
2.  按照本文开头的定义安装 MongoDB
3.  从配置服务器复制密钥文件，将其放在两个 shard 实例的用户主目录中
4.  给密钥文件适当的权限。我们需要在我们的分片集群中使用这个密钥文件，所以给它所需的权限，以便可以使用它

`sudo chown 400 /home/<username>/shardauthkey`

5.在任何文本编辑器中再次打开/etc/mongod.config 文件，并放置在文件底部的行下面。

```
sharding:
 clusterRole: shardsvr
replication:
 replSetName: shard1
net:
 bindIp: 0.0.0.0
```

对于 shard2，为了方便起见，只需将副本集的名称更改为 shard2

6.现在用下面的命令进入 mongod shell

`mongod --port=27017`

7.启动副本集。它是一个单节点副本集。如果你愿意，添加更多服务器到这个碎片

```
rs.initiate({
    _id: "shardsvr",
    members: [{
        _id: 0,
        host: "elastic_ip/public_ip/private_ip>:27018"
    }, ]
})
```

8.几秒钟后，你会看到 mongod 外壳变成主要的

9.运行以下命令并在此创建验证用户

```
use admin
db.createUser({
    user: "shard",
    pwd: "shard@123",
    roles: [{
        role: "root",
        db: "admin"
    }]
})
```

10.再次打开配置文件/etc/mongod.conf，并在文件底部添加以下行

```
security:
    keyfile: /home/ < username > /shardauthkey
```

11.我们还添加了 keyfile，它是从最初创建它的配置服务器上复制的

12.现在重启 mongod，用下面的命令进入 shell

`mongo --port=2707 -u shard -p shard@123 --authenticationDatabase=admin`

完成的

现在我们完成了初始设置。但主要问题是我们如何使每个节点能够相互通信。

# 网络调节

1.  配置服务器:config_server 需要直接访问端口 27017 上的<shar_1_ip>和端口 27017 上的<shard_2_ip>。</shard_2_ip></shar_1_ip>
2.  碎片服务器需要与配置服务器通信，因此它们需要访问端口 27017 上的

**AWS**

我将解释 AWS。您可以在任何云中以类似的方式进行设置

<shard_1_ip>:(可以是弹性 IP/内部 IP)使用 internal_ip 更安全</shard_1_ip>

<shard_2_ip>:(可以是弹性 IP/内部 IP)使用 internal_ip 更安全</shard_2_ip>

<config_server_ip>:(可以是弹性 IP/内部 IP)使用 internal_ip 获得更好的安全性</config_server_ip>

1.  在 AWS 上的 shard server 的安全组中，在端口 27017 上授予 config_server_ip 入站权限。
2.  在 AWS 上配置服务器的安全组中，将入站权限授予端口 27017 上的 shard_1_ip 和 shard_2_ip

在某些服务器上，您需要编辑防火墙规则。继续以同样的方式编辑

我们的集群已经准备好开始相互通信了

# 蒙古人:

1.  Ssh 到配置实例
2.  在/etc/中创建 mongos.conf 文件

```
systemLog:
 destination: file
 logAppend: true
 path: <log file directory>
net:
 port: 27020
 bindIp: 0.0.0.0
processManagement:
 timeZoneInfo: /usr/share/zoneinfo
sharding:
 configDB: config/<inter/public/elasticip/>:27017
security:
 keyFile: /home/<username>/shardauthkey
```

3.现在运行 mongos

`mongos —config /etc/mongos.conf`

4.为了不间断地运行 mongos，我们需要像 mongod 一样守护 mongos。在最新的 ubuntu 版本中，可以为 mongos 添加 systemctl 服务。或者，如果您想使用，也可以使用 supervisor。

5.现在使用下面的命令登录 mongos。我们已经在 mongo 命令中更改了端口，因为配置服务器运行在端口 27017 上，而 mongos 运行在端口 27020 上

`mongo --port=27020 -u shard -p shard@123 --authenticationDatabase=admin`

6.一旦你进入 mongo shell，你就可以添加碎片到集群了。

```
sh.addShard( "shard_1<public/private/elasticip>:27017")
sh.addShard( "shard_2<public/private/elasticip>:27017")
```

7.添加一个新用户来提供对 mongodb 客户端的访问

```
use admindb.createUser({ user: "sharduser",
  pwd: "sharduser@123",
  roles: [ { role: "root", db: "admin" } ]})
```

8.完成我们已经设置了集群

9.我们可以在 mongos 中启动一些分片收集，如果我们已经有了它们，否则我们可以在以后添加它们

```
sh.enableSharding("<database_name>")
sh.shardCollection("<database_name>.<collection_name>", {
    < shard key field >: "hashed"
})
```

# 碎片密钥

小心碎片键，因为它们提供了关于数据在整个集群中分布方式的信息。分片键还对数据更新和查询集合施加了一些限制

有关更多详细信息，请查看下面的文章:

[https://docs.mongodb.com/manual/core/sharding-shard-key/](https://docs.mongodb.com/manual/core/sharding-shard-key/)

# 使用 mongos

我们已经在配置服务器中设置了 mongos。转到其安全组，将 config_server 公共 IP 上的入站规则和端口 27020 分配给 all。

现在，您可以通过下面的命令使用分片集群

`mongo —host <public_ip_of_config_server> —port 27020 -u sharduser -p sharduser@123 --authenticationDatabase=admin`

在任何 mongo 客户端库中使用它。您可以将一个集群扩展到每台服务器 3 个节点的副本集，同样也可以扩展到多个 mongos。

我希望这能澄清你对 MongoDB shard 集群设置的疑虑。

感谢阅读