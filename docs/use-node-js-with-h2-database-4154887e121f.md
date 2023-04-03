# 将 Node.js 与 H2 数据库连接

> 原文：<https://medium.com/analytics-vidhya/use-node-js-with-h2-database-4154887e121f?source=collection_archive---------8----------------------->

![](img/93121ee726271d309aff6403a9d7c587.png)

*tldr:*下面的“解决方案”部分

# 动机

与我合作的 Java 编程团队正在使用 H2 数据库，因为它非常适合 Java(嵌入式，Java 速度快，整个数据库可以作为一个简单的文件轻松迁移)。他们必须使用存储在 H2 数据库中的数据来完成一项任务，但是他们还有其他几项同样重要的任务，所以他们需要一些额外的人手。

我专攻 Javascript 技术(React，Node.js)，很久以前就接触过 Java，但并不广泛。因此，我们决定另一项任务将在 Node.js 中完成，他们将 H2 数据库文件和任务的详细描述交给了我。

当时我意识到，这甚至都不能保证是可能的。所以我决定我的第一步应该是用这个 H2 数据库文件连接 Node.js 应用程序。如果这是不可能的，我们需要去寻找一些替代方案。

幸运的是，随着更流行的*嵌入式模式*的运行，H2 也提供了*客户机-服务器模式*。因此，我可以使用一个单独的数据库客户机，通过 TCP/IP 使用 JDBC 与 H2 数据库服务器通信。比嵌入式解决方案稍慢，但它是可行的。现在，如果 Node.js 没有一个用于 JDBC 通信的库，我就完了。幸运的是，Node.js 正好有一个 JDBC 包装器库，你只需要安装 Java，从官方页面下载 H2 驱动源代码，把它构建到。jar 文件，并按照 JDBC 包装库主页上的描述设置代码。并设置 Java 驱动的 H2 数据库服务器。

总的来说，一天后我确实设法完成了整个流程。Yaaay:)。

Node.js JDBC 包装器库的开发体验很差(没有 Promise 支持，没有 Typescript 支持，缺少文档，社区很小)。此外，我不想用 Java、数据库客户机驱动程序和 H2 数据库服务器来膨胀我的 PC。一定有更好的，更模块化的方法。

又过了一天，我找到了理想的装置。H2 数据库服务器肯定需要 Java，所以我决定把它放在 Docker 容器中。我发现这个不错的码头图片[https://hub.docker.com/r/oscarfonts/h2](https://hub.docker.com/r/oscarfonts/h2)，但我将不得不修改它一点。H2 数据库服务器，除了 JDBC 协议还支持 Postgresql 协议。它是默认启用的，或者带有以下标志:

```
-pg -pgAllowOthers -pgPort <PORT>
```

不幸的是，链接的 docker 映像被配置为不启动这个 Postgres 接口。所以我修改了它，创建了我自己的映像，我还将 H2 数据库服务器更新到了最新版本。下面是更新后的图片:[https://hub . docker . com/repository/docker/croraf/H2-PostgreSQL-server](https://hub.docker.com/repository/docker/croraf/h2-postgresql-server)

# 解决方案

提取图像，创建 Docker 容器并启动它:

```
$ docker pull croraf/h2-postgresql-server:1.0$ docker create -p 1521:1521 -p 81:81 -v /path/to/local/data_dir:/opt/h2-data --name=H2dbWithPostgres croraf/h2-postgresql-server:1.0$ docker start H2dbWithPostgres
```

这也将主机中的文件夹`/path/to/local/data_dir`绑定到容器中的文件夹`/opt/h2-data`。在图像中，`/opt/h2-data`被配置为 H2 数据库服务器保存数据库的文件夹。所以，要使用`mydatabase.mv.db`数据库文件只需将它放入`/path/to/local/data_dir`主机上即可。

现在，我们可以在 Node.js 应用程序中使用流行、简单且实用的 Postgres 客户端库[https://www.npmjs.com/package/pg](https://www.npmjs.com/package/pg)。

```
import { Client } from 'pg';const client = new Client({
  user: 'sa',
  host: 'localhost',
  database: 'mydatabase',
  password: '',
  port: 1521,
});const run = async () => {
  try {
    await client.connect();
    console.log('connected'); const res = await client.query('SELECT * FROM my_table');
    console.log(res);
  } catch (err) {
    console.log(err);
  } finally {
    await client.end();
  }
};run();
```