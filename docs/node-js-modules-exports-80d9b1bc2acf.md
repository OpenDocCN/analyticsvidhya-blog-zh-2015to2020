# 节点。JS 模块和导出

> 原文：<https://medium.com/analytics-vidhya/node-js-modules-exports-80d9b1bc2acf?source=collection_archive---------15----------------------->

![](img/9f4bf76dd790698251a2a6d18622cc4d.png)

[来源](https://commons.wikimedia.org/wiki/File:Node.js_logo.svg)

# 什么是模块？

NodeJS 中的一个模块是一个简单或复杂的 **JavaScript 文件**，它可以在你的整个 web 应用中使用。简而言之，每个**模块只是一个包含 JavaScript 代码的文件。**

## 有三种类型的模块:

## 核心模块:

这些类型的模块在 Node.JS 中包含最少的功能。常见的示例有:

**http** :构建简单的 http 服务器
**路径**:处理文件路径
**fs** :处理简单的文件输入/输出操作

## 本地模块

这些是你构建的模块:程序员。

编写模块是一个很好的实践，因为它促进了代码的整洁，并使代码模块化(你可以在其他项目中重用你的模块而不需要额外的努力)

## 第三方模块

这些类型的模块是由编程社区构建的。它们大多包含复杂的代码，这样程序员就可以在自己的应用程序中添加功能，而无需自己编写。这样，他也节省了时间。

# 我如何使用这些模块？

当使用模块时，你只需使用函数 *require()*

```
const myModule = require("moduleName")
```

## 示例:

让我们以使用 *fs* (文件系统模块)读取一个简单的文本文件为例:

打开记事本，在上面写下任何一段文字，然后保存在任何地方。姑且命名为“ *demofile.txt* ”。打开代码编辑器，编写这段代码，并将其保存到与。txt 文件:

```
const fs = require("fs")
fs.readFile("demofile.txt" , {encoding : 'utf-8'} , function(err,data) {
if(err) console.log(err)
console.log(data)
})
```

现在将它保存为" *fsExample.js* "并在命令行上运行它

```
node fsExample
```

(关于通过命令行运行代码的更多细节，我已经写了一个[指南](/@hussainarifkl/launching-node-js-scripts-from-command-line-173a1e85a615)

结果，您将看到文本文件的内容被打印出来

*readFile* 方法的详细信息:

**第一个参数**是您希望读取的文本文件的名称

**第二个参数**是你的**选项:**这里的编码方式是‘utf-8’，这样文本就可以打印出来了。如果你**不**指定选项，它将打印出缓冲值，人类无法读取。

**第三个参数**是回调函数。
什么是回调函数？在这里了解他们[。
首先，我们检查是否有错误(可能是读取文件时出错或其他原因)。)然后如果它们存在就打印出来，否则如果一切正常就打印出文件的内容。](https://developer.mozilla.org/en-US/docs/Glossary/Callback_function)

[fs 文档](https://nodejs.org/api/fs.html#fs_fs_readfile_path_options_callback)

# 使用第三方模块

首先通过 npm 安装一个模块

让我们以安装 *express* 为例

```
npm i express
```

然后*要求()*像这样:

```
const express = require('express')
```

注:我已经写了一个 NPM 的基本指南，见[这里](/@hussainarifkl/the-basics-of-npm-a32ee1d79901?sk=5aebd80f871bdc4ca990565e9730dc58)

# 使用本地模块

要使用你自己的模块(**本地模块**，你必须先将它*导出*。您可以导出**字符串、函数、变量**等。

语法如下:

```
module.exports = <Object to be exported> 
```

最简单的例子

创建一个名为' *message.js'* 的模块和一个名为' *app.js'* 的模块。

*message.js*

```
**module.exports** ="Hello World"
```

*app.js*

```
const message = **require**('./message.js')
console.log(message)
```

在命令行上运行代码 *app.js* 。因此，“Hello World”将被打印出来。

要导出一个**函数**，代码类似:

*message.js*

```
**module.exports** = function() {
console.log("Hello World")
}
```

*app.js*

```
const message = **require**("./message.js")
**message**();
```

# 导出对象

***导出*** 是 ***对象。*** 因此，你可以给它添加属性。*模块
模块*

*message.js*

```
**module.exports.sayHello** = function() {
console.log('Hello')
}**module.exports.sayCustomMessage** = function(message) {
console.log(message)
}
**module.exports.myVariable** = 4
**module.exports** = {
firstName: 'Hussain' 
lastName: 'Arif'
}
```

*app.js*

```
const message = **require**('./message.js')
message.*sayHello*()
message.*sayCustomMessage*('Goodbye!');
console.log(message.myVariable)
console.log(message.firstName + ' ' + message.lastName)
```

总之，要使用本地模块:
1)使用 **module.exports** 来导出它
2)使用 **require()** 来使用它的功能。

## 所需方法可用于导入:

```
const filesystem = **require**('fs') // core moduleconst express = **require**('express') // npm moduleconst server = **require**('./boot/server.js') // server.js file with a relative path down the treeconst server = **require**('../boot/server.js') // server.js file with a relative path up the treeconst server = **require**('/var/www/app/boot/server.js') // server.js file with an absolute pathconst server = **require**('./boot/server') // file if there's the server.js fileconst routes = **require**('../routes') // index.js inside routes folder if there's no routes.js fileconst databaseConfigs = **require**('./configs/database.json') // JSON file
```

**注意:要求第三方/NPM 模块不包括“.”还是“..”在你的参数里。这些模块已经存在于你的项目**的“节点模块”目录中

今天到此为止。回头见！

呆在家里，拯救生命。