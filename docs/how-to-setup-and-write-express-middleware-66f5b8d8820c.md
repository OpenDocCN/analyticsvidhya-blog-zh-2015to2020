# 如何设置和编写 Express 中间件

> 原文：<https://medium.com/analytics-vidhya/how-to-setup-and-write-express-middleware-66f5b8d8820c?source=collection_archive---------16----------------------->

# 中间件(Express，Nodejs):

![](img/1d6f693486612888915f52152045df39.png)

照片由[诺德伍德主题](https://unsplash.com/@nordwood?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 概述:

中间件是一个经常被误解的话题，因为它听起来似乎非常复杂，但实际上，中间件实际上非常简单。

中间件的整个思想是在发送响应的控制器动作之前和服务器从客户机获得请求之后执行一些代码。本质上，它是在请求过程中执行的代码，因此被命名为中间件。

不过，在深入了解中间件的细节之前，我想设置一个具有两条路径的基本 Express 服务器。

# 设置快速服务器(Nodejs):

要开始使用 Node.js 项目，您需要运行几个简单的命令，您可以遵循下面提到的简单步骤:

**步骤:**

1.  运行 npm init -y 来创建基本的 package.json 文件。
2.  通过运行安装 Express，npm i express。
3.  使用以下代码创建一个 server.js 文件。

```
const express = require('express')
const app = express()app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', (req, res) => {
  res.send('Users Page')
})app.listen(3000, () => console.log('Server Started'))
```

这个 server.js 文件只是在端口 3000 上设置了一个服务器，它有两个路由，一个主页路由和一个用户页面路由。

4.运行 node server.js，在控制台中看到一条消息说**服务器启动了。**

5.运行 node server.js，在控制台中看到一条消息说**服务器启动了。**

6.打开任何浏览器到 localhost:3000，见消息**主页**。

7.转到 localhost:3000/users，您应该会看到消息 **Users Page** 。

这就是我们在本文余下部分需要的所有基本设置。当我们进行更改时，您需要在控制台中重新启动服务器，以使更改生效。

# 什么是中间件？

我简要介绍了中间件，它是在服务器接收请求之后、控制器动作发送响应之前执行的功能，但是还有一些特定于中间件的功能。

最重要的是，中间件功能可以访问响应(`res`)和请求(`req`)变量，并可以根据需要修改或使用它们。中间件函数还有第三个参数，这是一个`next`函数。这个函数很重要，因为必须从一个中间件调用它，才能执行下一个中间件。如果这个函数没有被调用，那么包括控制器动作在内的其他中间件都不会被调用。

仅仅从文本上理解这些有点困难，所以在下一节中，我们将创建一个日志中间件，它将记录用户请求的 URL。

# 如何创建日志中间件

正如我在上一节提到的，中间件有三个参数，`req`、`res`和`next`，所以为了创建中间件，我们需要创建一个有这三个输入的函数。

```
const express = require('express')
const app = express()app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', (req, res) => {
  res.send('Users Page')
})function loggingMiddleware(req, res, next) {  console.log('Inside Middleware')}
app.listen(3000, () => console.log('Server Started'))
```

我们现在有了一个用一些占位符内容定义的基本中间件函数的外壳，但是应用程序没有使用它。Express 有几种不同的方式可以定义要使用的中间件，但是对于这个例子，我们将通过将它添加到应用程序级别，使这个中间件在每个控制器动作之前执行。这可以通过像这样对`app`变量使用`use`函数来完成。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)
app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', (req, res) => {
  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log('Inside Middleware')
}app.listen(3000, () => console.log('Server Started'))
```

应用程序现在使用我们定义的中间件，如果我们重启服务器并导航到应用程序中的任何页面，您会注意到在控制台中出现消息 **Inside Middleware** 。这很好，但是有一个小问题。应用程序现在永远加载，永远不会真正完成请求。这是因为在我们的中间件中，我们不调用`next`函数，所以控制器动作永远不会被调用。我们可以在登录后调用`next`来解决这个问题。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', (req, res) => {
  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log('Inside Middleware')
  next()}app.listen(3000, () => console.log('Server Started'))
```

现在，如果您重新启动服务器，您会注意到一切都记录正确，网页加载正常。接下来要做的是实际注销用户在中间件内部访问的 URL。这就是`req`变量派上用场的地方。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', (req, res) => {
  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log(`${new Date().toISOString()}: ${req.originalUrl}`)  next()
}app.listen(3000, () => console.log('Server Started'))
```

日志中间件现在在应用程序的所有路径上都 100%正确地工作，但是我们仅仅触及了中间件有用性的表面。在下一个例子中，我们将看看如何为用户页面创建一个简单的授权中间件。

# 高级中间件示例

首先，我们需要创建另一个函数作为中间件。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', (req, res) => {
  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log(`${new Date().toISOString()}: ${req.originalUrl}`)
  next()
}function authorizeUsersAccess(req, res, next) {  console.log('authorizeUsersAccess Middleware')  next()}
app.listen(3000, () => console.log('Server Started'))
```

这只是一个用作中间件的函数外壳，但是我们现在可以将它添加到我们的用户页面路由中，以确保我们的中间件只在用户页面路由中执行。这可以通过将该函数作为参数添加到用户页面的`app.get`函数中来实现。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', authorizeUsersAccess, (req, res) => {  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log(`${new Date().toISOString()}: ${req.originalUrl}`)
  next()
}function authorizeUsersAccess(req, res, next) {
  console.log('authorizeUsersAccess Middleware')
  next()
}app.listen(3000, () => console.log('Server Started'))
```

现在，如果您重启服务器并转到用户页面，您应该会看到消息**authorizeUsersAccess Middleware**，但是如果您转到主页，该消息将不会显示。我们现在有了只在应用程序中的单一路径上执行的中间件。接下来要做的是填充这个函数的逻辑，这样如果用户没有访问页面的权限，他们将会得到一个错误消息。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', authorizeUsersAccess, (req, res) => {
  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log(`${new Date().toISOString()}: ${req.originalUrl}`)
  next()
}function authorizeUsersAccess(req, res, next) {
  if (req.query.admin === 'true') {    next()  } else {    res.send('ERROR: You must be an admin')  }}app.listen(3000, () => console.log('Server Started'))
```

这个中间件现在检查查询参数`admin=true`是否在 URL 中，如果不在，则向用户显示错误消息。您可以转到`http://localhost:3000/users`进行测试，您会看到一条错误消息，说明您不是管理员。如果你转而去`http://localhost:3000/users?admin=true`，你将被发送到普通用户页面，因为你将 admin 的查询参数设置为 true。

中间件真正有用的另一件事是在中间件之间发送数据的能力。下一个函数无法做到这一点，但是您可以修改`req`或`res`变量来设置您自己的定制数据。例如，在前面的示例中，如果用户是管理员，我们想要将变量设置为 true，我们可以很容易地做到这一点。

```
const express = require('express')
const app = express()app.use(loggingMiddleware)app.get('/', (req, res) => {
  res.send('Home Page')
})app.get('/users', authorizeUsersAccess, (req, res) => {
  console.log(req.admin)  res.send('Users Page')
})function loggingMiddleware(req, res, next) {
  console.log(`${new Date().toISOString()}: ${req.originalUrl}`)
  next()
}function authorizeUsersAccess(req, res, next) {
  if (req.query.admin === 'true') {
    req.admin = true    next()
  } else {
    res.send('ERROR: You must be an admin')
  }
}app.listen(3000, () => console.log('Server Started'))
```

这段代码在`req`对象上设置一个管理变量，然后在用户页面的控制器动作中访问该变量。

# 中间件附加信息

这是你需要知道的关于中间件功能的大部分事情，但是还有一些额外的事情需要知道。

# 1.控制器动作就像中间件一样

你可能已经注意到的一件事是，有一个`req`和`res`变量的控制器动作非常类似于中间件。这是因为它们本质上是中间件，但是在它们之后没有其他中间件。它们是链的末端，这就是为什么在控制器动作中从来没有下一个调用。

# 2.调用 next 不同于调用 return

到目前为止，我看到开发人员在使用中间件时犯的最大错误是，他们将`next`函数视为已经退出了中间件。以这个中间件为例。

```
function middleware(req, res, next) {
  if (req.valid) {
    next()
  }
  res.send('Invalid Request')
}
```

从表面上看，这段代码似乎是正确的。如果请求有效，则调用`next`函数，如果无效，则发送错误消息。问题是`next`函数实际上并没有从中间件函数返回。这意味着当`next`被调用时，下一个中间件将被执行，这将一直持续到不再有中间件被执行。然后在所有的中间件之后，在这个中间件完成执行之后，代码将在每个中间件中的`next`调用之后立即恢复。这意味着在这个中间件中，错误消息将总是被发送给用户，这显然不是您想要的。防止这种情况的一个简单方法是当你调用`next`时简单地返回

```
function middleware(req, res, next) {
  if (req.valid) {
    return next()  }
  res.send('Invalid Request')
}
```

现在代码在调用`next`后将不再执行，因为它将从函数中返回。查看这个问题的一个简单方法是使用下面的代码。

```
const express = require('express')
const app = express()app.get('/', middleware, (req, res) => {
  console.log('Inside Home Page')
  res.send('Home Page')
})function middleware(req, res, next) {
  console.log('Before Next')
  next()
  console.log('After Next')
}app.listen(3000, () => console.log('Server Started'))
```

当您运行此代码并转到主页时，控制台将按顺序打印出以下消息。

```
Before Next
Inside Home Page
After Next
```

实际上，中间件被调用，并注销 before 语句。然后调用 next，因此调用下一组中间件，这是记录主页消息的控制器动作。最后，控制器动作结束执行，因此中间件在`next`之后执行代码，这将注销 after 语句。

# 3.中间件将按顺序执行

这似乎不言自明，但是当你定义中间件时，它将按照使用的顺序执行。以下面的代码为例。

```
const express = require('express')
const app = express()app.use(middlewareThree)
app.use(middlewareOne)app.get('/', middlewareTwo, middlewareFour, (req, res) => {
  console.log('Inside Home Page')
  res.send('Home Page')
})function middlewareOne(req, res, next) {
  console.log('Middleware One')
  next()
}function middlewareTwo(req, res, next) {
  console.log('Middleware Two')
  next()
}function middlewareThree(req, res, next) {
  console.log('Middleware Three')
  next()
}function middlewareFour(req, res, next) {
  console.log('Middleware Four')
  next()
}app.listen(3000, () => console.log('Server Started'))
```

由于`app.use`语句最先出现，这些语句中的中间件将按照它们被添加的顺序首先执行。接下来定义`app.get`中间件，它们将再次按照在`app.get`函数中的顺序执行。如果运行，这将导致以下控制台输出。

```
Middleware Three
Middleware One
Middleware Two
Middleware Four
```

# 结论

这就是关于中间件的所有知识。中间件在清理代码和使用户授权和认证等事情变得更加容易方面非常强大，但由于中间件令人难以置信的灵活性，它的用途远不止于此。

这是我这边的人说的，我想现在你可以用中间件了。如果你喜欢它，那么请随时点击拍手，跟随按钮和反馈是最受欢迎的。

*谢谢大家，让我们补上新的。*