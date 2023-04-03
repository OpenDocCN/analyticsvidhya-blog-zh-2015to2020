# MySQL river for elastic search | go-MySQL-elastic search in Ubuntu 18.04

> 原文：<https://medium.com/analytics-vidhya/mysql-river-for-elasticsearch-go-mysql-elasticsearch-in-ubuntu-18-04-771e2c172c4f?source=collection_archive---------11----------------------->

![](img/b1ecf9c0a5e5dec8eea4ae6eeaf8a0e5.png)

这篇博文分享了我为 ElasticSearch 建立 MySQL 河流的经验。

> river 是一种简单的方法，可以为您的 Elasticsearch 数据仓库建立一个连续的数据流。这是一个在 Elasticsearch 集群中运行的插件服务，它提取数据(或推送数据)，然后在集群中进行索引。
> 它比传统的手动数据索引方法更实用，因为一旦配置完毕，所有数据都会自动更新。这降低了复杂性，也使得实时系统的创建成为可能。

# **背景和动机**

在这个数据驱动的时代，无论您是构建小规模还是大规模的应用程序，其中一个主要因素就是数据。根据数据的性质和开发人员的选择，这些非常重要的 blobs 将存放在 NoSQL 或 SQL 数据库中。

最近，我和我的同事正在开发一个评分系统，包括基于不同标准的用户评分。由于我们的数据表现出巨大的关系特征，我们选择了 MySQL 服务器来存储它们。经过一些头脑风暴和想法分享，我们能够设计出一个与我们的数据和需求相一致的有效模式。

在实现了算法和其他支持数据管道之后，我们对其进行了一些基准测试。
我们得出的一个推论是，我们的引擎需要自动更正/补全功能，因为来自广泛用户群的数据往往会有错别字、缩写等。我们选择了 Elasticsearch *(一个存储和查询巨大文本数据集的伟大工具)*来创建搜索空间。

> Elasticsearch 是一个灵活、强大的开源、分布式实时搜索和分析引擎。它速度快，易于使用，在许多情况下非常有用。

为了实现这样的搜索空间，有必要将数据从 MySQL 数据库传输到 es 索引。MySQL 转储第一次可以用于播种索引，但是真正的挑战是数据很容易改变，所以我们需要在 MySQL 数据库和 es 索引之间建立连续的数据流。

因此，我们开始在广阔的万维网中寻找这样的机制，我们遇到了一些真正令人难以置信的项目。其中一个是我们自己的 ES stack 的 [*logstash*](https://www.elastic.co/blog/how-to-keep-elasticsearch-synchronized-with-a-relational-database-using-logstash) *，*虽然看起来很有希望，但它不能满足我们所有的需求。然后我们遇到了这个用 Go 构建的很棒的服务。[https://github.com/siddontang/go-mysql-elasticsearch](https://github.com/siddontang/go-mysql-elasticsearch)

> go-mysql-elasticsearch 是一项将你的 mysql 数据自动同步到 elasticsearch 的服务。
> 首先用`[mysqldump](https://dev.mysql.com/doc/refman/8.0/en/mysqldump.html)`取原始数据，然后用 [binlog](https://dev.mysql.com/doc/refman/8.0/en/binary-log.html) 增量同步数据。

所以，现在让我们进入正题，设置这个。

# 先决条件

*   [Go (1.9+)](https://golang.org/doc/install)
*   [MySQL ( < 8.0)](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/) *【本地或云实例】*
*   [弹性搜索(< 6.0)](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)

> *如果您的环境没有配置这些要求，请遵循以上链接获取安装说明。*

尽管开发者建议使用 MySQL <8.0 and ES <6.0, we implemented it with MySQL 5.7 and ES 7.7 and it worked without any issues.

# Installing MySQL river

Run the following command from your terminal to fetch the project

```
*go get github.com/siddontang/go-mysql-elasticsearch*
```

> *注意:它会显示一些信息，忽略它。
> 默认下载位置将是$HOME/go，如果您想要使用自定义位置，请在您的环境中使用*[*GOPATH*](https://golang.org/doc/gopath_code.html)*变量对其进行配置。* `*Add the following line to your ~/.bashrc file*`

现在项目将被下载并安装到`$GOPATH`中，它包含目录`src/github.com/siddontang/go-mysql-elasticsearch/`，以及那些库及其依赖项的编译包(在`pkg/`)。

更改到项目目录

```
cd $HOME/go/src/github.com/siddontang/go-mysql-elasticsearch-- or --cd $GOPATH/src/github.com/siddontang/go-mysql-elasticsearch
```

让我们构建程序的可执行文件，在您的终端中运行以下命令

```
make
```

# 使用 MySQL river

## 配置 MySQL

*   [配置你的 MySQL *binlog* 使用行格式](https://dev.mysql.com/doc/refman/5.7/en/binary-log-setting.html)

从 mysql shell 中运行以下命令

```
SET GLOBAL binlog_format = 'ROW';-- or -- For individual clients use the below commandSET SESSION binlog_format = ‘ROW’;
```

*   创建你的 MySQL 数据库和表格。

> *注意:将被同步的 MySQL 表应该有一个主键，也允许多列主键。*

## 配置弹性搜索

*   如果可能，创建相关的 Elasticsearch 索引、文档类型和映射，否则，Elasticsearch 将自动创建这些内容。

> *注意:如果没有配置 ES 索引，将使用默认映射。我们的方法需要精确的搜索结果，所以我们创建了带有自定义映射的索引。*

## 配置河流

mysql-es 的配置在
*$ GOPATH/src/github . com/siddontang/go-MySQL-elastic search/etc*下的 ***river.toml*** 文件中提供

将工作目录更改为

```
cd *$GOPATH/src/github.com/siddontang/go-mysql-elasticsearch/etc*
```

在您喜欢的编辑器中打开 **river.toml** config

```
nano river.toml
```

***MySQL 配置***

```
# MySQL address, user and password
# user must have replication privilege in MySQL.my_addr = "127.0.0.1:3306"
my_user = "root"
my_pass = ""
my_charset = "utf8" 
```

您也可以使用云托管的 mysql 服务，如 RDS 等..现在用 *flavor* param 指定您选择的 db。

```
# mysql or mariadb
flavor = "mysql"
```

“go-mysql-elasticsearch 首先使用 mysqldump 获取原始数据，然后使用 binlog 增量同步数据”,因此，如果您也想索引现有数据，则使用以下配置指定转储位置，否则，只需对其进行注释或留空。

```
# mysqldump execution path
# if not set or empty, ignore mysqldump.mysqldump = "./var/mysqldump"
```

> 注意:我们在尝试从 RDS 恢复转储时遇到了一些问题，所以如果您正在使用 RDS 实例，请查看本帖的**获取部分**以了解有关此问题的更多信息。

***ES 配置***

```
# Set true when elasticsearch use https
#es_https = false
# Elasticsearch addresses_addr = "127.0.0.1:9200"# Elasticsearch user and password, maybe set by shield, nginx, or x-packes_user = ""
es_pass = ""
```

***数据库配置***

假设我们有一个名为 dbx 的数据库，其中包含表 tba、tbb、tbc …为了同步该数据库及其首选表，我们需要在 ***源配置*** 下指定它们

```
# MySQL data source
[[source]]
schema = "dbx"
tables = ["tba", "tbb"]
```

如果您想同步数据库中的所有表，只需使用通配符' * '。

```
# MySQL data source
[[source]]
schema = "dbx"
tables = ["*"]
```

也支持其他正则表达式，你可以做类似的事情来同步相似格式的表

```
# MySQL data source
[[source]]
schema = "dbx"
tables = ["tb[a-z]"]
```

如果您想要同步多个数据库，您可以用它们各自的数据库名称复制这个配置。因此，同步 dbx 和 dby 的配置如下所示

```
# MySQL data source
[[source]]
schema = "dbx"
tables = ["tb[a-z]"][[source]]
schema = "dby"
tables = ["tbc", "tbd"]
```

***指标配置***

比方说，您希望将表 tba 同步到索引 tb_index。这是在规则部分下配置的。

```
[[rule]]
schema = "dbx"
table = "tba"
index = "tb_index"
type = "tba"
```

> *注意:默认索引和类型名将是表名的索引和类型名。根据你的需要改变它们。*

如果您只想同步表中的特定列，请使用下面的配置

```
[[rule]] 
schema = "dbx"
table = "tba"
index = "tb_index"
type = "tba" 

filter = ["col_a"] 

[rule.field] 
col_a="name"
```

这里 ***过滤器*** 表示要同步到索引中的列，而 ***规则.字段*** 表示列与索引的映射关系。*即*在这种情况下，来自*‘col _ a’*的值将被映射到 tb_index 的【T53’‘name’字段中。

***规则字段*** 配置也支持数据类型转换/规范，

> *这将把 col_a 列映射到弹性搜索名称*

```
[rule.field]
 col_a="name"
```

> *这将把 col_a 列映射到弹性搜索名称，并使用数组类型*

```
[rule.field] 
col_a="name,list"
```

> *这将把列 col_a 映射到弹性搜索 col_a，并使用数组类型*

```
[rule.field] 
col_a=",list"
```

> *“list”修饰符将 mysql 字符串字段如“a，b，c”转换为弹性数组类型“{ a”，“b”，“c”} ”,如果您需要在弹性搜索中使用这些字段进行过滤，这将特别有用。*
> 
> *如果 created_time 字段类型为“int ”,而您想在 ES 中将其转换为“date”类型，您可以按如下方式操作*

```
[rule.field] 
created_time=",date"
```

我们发现非常有用的另一个特性是在表规范中支持通配符。当您想要索引一个表的分块部分时，它特别有用。

```
[[rule]] 
schema = "dbx"
table = "tb[a-z]"
index = "tb_index"
type = "tba" 

filter = ["col_a"] 

[rule.field] 
col_a="name"
```

> *注意:确保匹配给定通配符的表应该具有相同的模式*

为了将多个表同步到不同的索引中，只需复制带有相应表和索引名称的规则配置，就像我们对多个数据库所做的那样。

因此，为了将 tba 和 tbd 同步到 2 个索引中，比如 tb_index_1 和 tb_index_2，配置将是

```
[[rule]] 
schema = "dbx"
table = "tba"
index = "tb_index_1"
type = "tba" 

filter = ["col_a"] 

[rule.field] 
col_a="name" [[rule]] 
schema = "dby"
table = "tbd"
index = "tb_index_2"
type = "tbd" 

filter = ["col_d"] 

[rule.field] 
col_d="name"
```

通过这些配置，我们成功地建立了一个 river，用于将 mysql 数据同步到 ES 中。

> *go-mysql-elasticsearch 还提供了更多的配置，你可以在项目*[*R*](https://github.com/siddontang/go-mysql-elasticsearch/blob/master/README.md)*eadme . MD 文件:* [https://github.com/siddontang/go-mysql-elasticsearch#source](https://github.com/siddontang/go-mysql-elasticsearch#source)中查看

完成了。现在让我们开始我们的河流。

## 部署河流

cd 到项目目录中

```
cd *$GOPATH/src/github.com/siddontang/go-mysql-elasticsearch/*
```

通过运行可执行文件来启动河流

```
./bin/go-mysql-elasticsearch -config=./etc/river.toml
```

如果您能够成功地配置河流，您将会发现，对于表中所做的每一项更改，您的数据都会流入相应的索引中。速度极快，而且是实时的。

# 将河流作为一种服务来运营

如果您希望 river 在后台运行，那么在您的 Linux 环境中，在 */etc/systemd/system* 中创建一个系统服务文件

创建服务文件

```
sudo nano /etc/systemd/system/go-mysql-es-river.service
```

将下列行添加到服务文件中

```
[Unit] 
Description=Instance to serve go mysql es river 
After=network.target 

[Service] 
User=ubuntu 
Group=www-data 
WorkingDirectory=$GOPATH/src/github.com/siddontang/go-mysql-elasticsearch 
ExecStart=$GOPATH/src/github.com/siddontang/go-mysql-elasticsearch/bin/go-mysql-elasticsearch -config=/
$GOPATH/src/github.com/siddontang/go-mysql-elasticsearch/etc/river.toml 

[Install] 
WantedBy=multi-user.target
```

保存文件 *(Ctrl+o &回车)并退出(Ctrl+x)*

现在让我们开始服务吧

```
sudo systemctl start go-mysql-es-river.service
```

如果您希望 river 在系统启动时启动，那么使用

```
sudo systemctl enable go-mysql-es-river.service
```

# 小外卖

我们在从 RDS 同步 mysqldump 时遇到了一个问题。该问题与 go-mysql-elasticsearch 无关，而是与 RDS 相关的权限错误。

> **无法执行“用读锁刷新表”:用户“root”@“%”的访问被拒绝(使用密码:是)(1045)**

这是因为没有足够的权限运行带有 AWS RDS 中的
-主数据标志的 mysqldump。

> ***如果你遇到这个错误，要么注释 river.toml 中的 mysqldump 配置，用 Index API 或者 Bulk API 填充 es 索引。***

如果没有，您可以按照下面的步骤使用 mysqldump 填充索引。

1.  使用 mysqldump 转储 RDS MySQL 服务器，并将其恢复到本地 MySQL db。
2.  关闭 RDS MySQL 服务器
3.  将`my_addr`设置为本地数据库，启动 go-mysql-elasticsearch，并等待搜索完成
4.  停止 go-mysql-elasticsearch 并删除 var/master.info
5.  重新启动 RDS MySQL 服务器
6.  重启 go-mysql-elasticsearch，用`my_addr`设置 RDS MySQL 服务器，用`mysqldump`清空

# 来自开发者的一些注释

*   binlog 格式必须是行。
*   对于 MySQL，二进制日志行图像必须是满的，如果在 MySQL 中使用最小的或无二进制日志行图像更新 PK 数据，您可能会丢失一些字段数据。MariaDB 仅支持整行图像。
*   运行时不能改变表格格式。
*   将要同步的 MySQL 表应该有一个 PK(主键)，现在允许多列 PK，例如，如果 PKs 是(a，b)，我们将使用“a:b”作为键。PK 数据将在 Elasticsearch 中用作“id”。您也可以用其他列配置 id 的组成部分。
*   你应该先在 Elasticsearch 中创建关联映射，我不认为使用默认映射是一个明智的决定，你必须知道如何精确搜索。
*   `mysqldump`必须与 go-mysql-elasticsearch 存在于同一个节点，如果不存在，go-mysql-elasticsearch 将尝试只同步 binlog。
*   不要在一个 SQL 中同时更改太多行。

# 结论

这项服务对我们的团队产生了巨大的影响，比如使我们的数据库实例和搜索空间之间的数据同步自动化变得更加顺畅，使数据传输实时发生，并跟踪所有必要的 CRUD 操作。最终，当我们测试我们的应用程序时，我们发现 SQL server 中所做的更改实时地反映在相应的 es 索引上。

# 参考

> [1。https://github.com/siddontang/go-mysql-elasticsearch](https://github.com/siddontang/go-mysql-elasticsearch)
> 2。[https://golang.org/doc/](https://golang.org/doc/)3
> 。[https://www . elastic . co/blog/how-to-keep-elastic search-synchronized-with-a-relational-database-using-log stash](https://www.elastic.co/blog/how-to-keep-elasticsearch-synchronized-with-a-relational-database-using-logstash)
> 4 .[https://dev.mysql.com/doc/refman/8.0/en/mysqldump.html](https://dev.mysql.com/doc/refman/8.0/en/mysqldump.html)5。[https://dev.mysql.com/doc/refman/8.0/en/binary-log.html](https://dev.mysql.com/doc/refman/8.0/en/binary-log.html)6
> 。[https://dev . MySQL . com/doc/ref man/5.7/en/binary-log-setting . html](https://dev.mysql.com/doc/refman/5.7/en/binary-log-setting.html)
> 7 .[https://dev . MySQL . com/doc/MySQL-installation-extract/5.7/en/](https://dev.mysql.com/doc/mysql-installation-excerpt/5.7/en/)
> 8 .[https://www . elastic . co/guide/en/elastic search/reference/current/install-elastic search . html](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html)