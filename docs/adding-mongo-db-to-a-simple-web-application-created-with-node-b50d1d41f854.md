# 将 Mongo DB 添加到用 Node 创建的简单 Web 应用程序中

> 原文：<https://medium.com/analytics-vidhya/adding-mongo-db-to-a-simple-web-application-created-with-node-b50d1d41f854?source=collection_archive---------27----------------------->

![](img/d522c57e8d778b2d64505b47862fe517.png)

摘自[此处](https://www.mongodb.com/brand-resources)

在本文中，我们将把数据库添加到我们简单的 publisher 创建的[这里](/@maheshsai252/a-simple-online-publisher-using-node-js-81de6d8f40ee)。

将对我们的 publisher web 应用程序进行以下修改:

1.  我们需要在数据库中存储从用户收到的信息。
2.  我们需要向用户显示信息。

## 为什么是 Mongo DB？

[Mongo DB](https://www.mongodb.com) 是 NoSQL 的数据库。现实世界的数据并不总是以表格的形式出现。所以，我们没有 sql 数据库。

Mongo DB 使用 JSON 文档来存储数据。它还支持正则表达式。

我们将在我们的网站中使用[猫鼬](https://www.npmjs.com/package/mongoose)。Mongoose 是一个工具，使我们的工作与 Mongo 数据库容易。

# 安装和导入猫鼬

使用以下命令安装

```
npm i mongoose
```

在 app.js 中导入

```
const mongoose = require("mongoose");
```

# 连接到数据库

我们正在创建一个博客数据库并连接到它。

将此内容包含在 app.js 中

```
mongoose.connect("mongodb://localhost:27017/blogDB", {useNewUrlParser: true, useUnifiedTopology: true});
```

# 在 app.js 中修改回调

我们将把 blogDataBase 添加到上一篇文章中创建的 publisher 中，而不改变其功能。

回想一下我们用来存储内容的 posts 数组。

```
let posts=[]
let post = {
    title : req.body.title,
    postc : req.body.post
  };
 posts.push(post);
```

现在我们将在我们的 blogDataBase 中创建 posts 集合。

首先，我们需要为 posts 集合创建模式

```
const postSchema = {title: String,content: String};
```

现在创建一个收藏

```
const Posts = mongoose.model("Posts", postSchema);
```

我们创造了收藏。当用户写文章时，我们会将文档添加到我们的集合中。修改“/撰写”路由的回调。

```
app.post("/compose",function(req,res){

  const post = new Posts ({ title: req.body.title, content: req.body.post });
 post.save();
 res.redirect("/");});
```

“post”是保存在 post 集合中的文档。

完成向数据库添加用户输入。现在，我们从主页和帖子页的数据库中检索数据。

主页回调修改如下

```
app.get("/",function(req,res)
{
  Post.find({}, function(err, posts){
   console.log(posts);
   res.render("home", {startingContent: homeStartingContent,posts: posts});
});
}
);
```

Post.find( <condition>，function(err， <result>){…})是在集合中查找文档的语法。</result></condition>

与此类似，findOne 函数可以根据条件从集合中检索文档。我们将使用这个内部 post 回调。

```
app.get("/posts/:postname",function(req,res){

 Post.findOne({title: req.params.postname}, function(err, post){
    if(!err)
    {
      res.render("post", {title: post.title,content: post.content});}
    else
    {
      console.log(err);
    }});
});
```

就是这样！！！

# 警告

在 mac 中使用“mongod”命令启动 mongoDB 时可能会出现错误。

请尝试以下命令

```
sudo mongod --dbpath /System/Volumes/Data/data/db
```

# 结论

无需修改 ejs 文件的任何代码，我们已经将数据库添加到我们在此创建的 publisher 网站。

完整的代码可以在[这里](https://github.com/maheshsai252/publisher)找到。