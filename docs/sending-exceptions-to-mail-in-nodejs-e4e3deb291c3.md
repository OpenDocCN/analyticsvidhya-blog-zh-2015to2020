# 向 NodeJS 中的邮件发送异常

> 原文：<https://medium.com/analytics-vidhya/sending-exceptions-to-mail-in-nodejs-e4e3deb291c3?source=collection_archive---------4----------------------->

*在编写和维护一个应用程序开发的过程中，我们会面临无数的 bug 和错误。在开发时解决这些相对容易。应用程序部署后，我们仍然可能会面临一些意想不到的异常。不管怎样，我们必须找到一种方法来处理这些异常。*

![](img/463df8fe0cdc6492708ef11ad4561f87.png)

有许多更好的方法来处理这些异常(例如 Sentry)。我们将使用一个不被推荐的捷径。我们将学习如何在 NodeJS 的帮助下将这些异常发送到邮件。

我们将使用一个名为 nodemailer 的邮件模块。Nodemailer 是 NodeJS 世界中最流行的模块，用于在服务器之间发送电子邮件。它使发送电子邮件变得非常容易。

让我们创建两个 js 文件。第一个文件将被命名为 **mail_server.js** 。我们将把所有配置代码放在这个文件中，以便能够发送电子邮件。另一个就叫 **app.js** 。在这个文件中，我们将编写一个 try catch 块，并使用 **mail_server.js** 作为内部模块向声明的电子邮件地址发送错误。

**创建文件**
让我们从创建一个文件开始，我们将在其中定义内容。创建一个新文件，命名为 **mail_server.js** 。正如我上面提到的，我们需要一个名为 nodemailer 的包。因此，让我们通过在终端中运行以下命令来安装该模块。
`npm install nodemailer`

现在，打开 **mail_server.js** ，将 nodemailer 包导入其中。为此，我们在 NodeJS 中使用 require 命令。
`var nodemailer = require(‘nodemailer’);`

下一步是创建一个包含发送电子邮件的所有配置的函数。我创建了一个名为 **mailfunc** 的函数，它传递一个名为 **errormessage** 的参数。

```
var nodemailer = require(‘nodemailer’);function mailfunc(errormessage){
  var transporter = nodemailer.createTransport({
    service: ‘gmail’,
    auth: {
      user: ‘youremail@gmail.com’,
      pass: ‘yourpassword’
    }
  });
  var mailOptions = {
    from: ‘youremail@gmail.com’,
    to: ‘someoneelsesmail@outlook.com’,
    subject: ‘Error Message’,
    text: errormessage,
  };
  transporter.sendMail(mailOptions, function (error, info) {
    if (error) {
      console.log(error);
    } else {
      console.log(‘Email sent: ‘ + info.response);
    }
 });
}
exports.mailfunc= mailfunc;
```

Gmail 将是我们在这个项目中使用的邮件服务器。要从我们的 NodeJS 应用程序等第三方应用程序使用 Gmail 作为服务发送电子邮件，我们必须在 Gmail 中启用“允许不太安全的应用程序”。点击[此处](https://myaccount.google.com/lesssecureapps)启用“允许不太安全的应用程序”。

还有一件事要添加到文件中，这就是导出我们在 **mail_server.js** 中创建的函数。供其他程序使用。
`exports.mailfunc= mailfunc;`

是时候为 **app.js** 编写一些代码，并导入我们刚刚导出的 **mail_server.js** 供其他程序像 npm 模块一样使用。我们需要做的就是像对其他模块一样要求它。
`const mail = require(‘./mail_server’);`

让我们将 try catch 块添加到我们的 **app.js** 中，看看它是否真的在工作。

```
const mail = require(‘./mail_server’);try {
  throw ‘myException’; // generates an exception
}
catch (e) {
  /*email exception object to specified mail address*/
  mail.mailfunc(e.toString());
  console.log(e);
}
```

如你所见，我们需要做的就是在 catch 块中调用 **mailfunc** 。仅此而已！！从现在开始，我们将通过电子邮件接收我们的应用程序错误，这些错误将在生产中显示出来。

感谢您的点击。