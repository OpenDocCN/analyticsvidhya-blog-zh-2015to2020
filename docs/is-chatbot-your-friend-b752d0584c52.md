# 聊天机器人是你的朋友吗？

> 原文：<https://medium.com/analytics-vidhya/is-chatbot-your-friend-b752d0584c52?source=collection_archive---------35----------------------->

如果你想参与一个有回应的对话，进入聊天机器人。

![](img/c0672be8a4c8da0e5eeaef16b976169d.png)

你可以随时来聊天机器人。聊天机器人是一个有保证的帮助代理，你总是会得到回应。即使通常回答可能没有帮助。自从新冠肺炎疫情以来，人们对在线获取信息的需求越来越大，聊天机器人就是一个利用聊天机器人功能接触大众的完美例子。聊天机器人使不太懂技术的人也能获得信息。有了口头命令和输出，聊天机器人就成了一个很好的可访问代理。

让我们继续学习如何在 React 应用程序中安装聊天机器人。

首先创建一个 React 应用程序

`npx create-react-app <coronavirus-chatbot>`

我们将使用一种简单的格式来处理聊天机器人的新冠肺炎症状，用交互的方式来检索关于症状的信息。我们将使用一个 [npm 包](https://lucasbassetti.com.br/react-simple-chatbot/)来使用聊天机器人。

第二步是安装聊天机器人组件；

`npm install react-simple-chatbot — save`

因为我们需要样式化的组件，我们也将安装它。

`npm install — save styled-components`

我们现在可以创建一个聊天机器人组件:

聊天机器人的工作方式是，steps 中的每个对象都有一个`id`、`message`和`trigger`。如果下一个对象的 id 与前一个对象的触发器匹配，那么它将触发下一个对象，显示下一个对象中发布的消息。

这是机器人将如何一步一步地进行。如果你想个性化机器人，你可以问一个问题。

例如:

```
{id: “Ask Name”,message: “What your name?”,trigger: “ name”},
```

一旦用户输入一个名字，机器人就可以将这个名字保存为`{previousValue}`，并在下一个提示中将其吐回。

```
{id: “name,message: “Hi {previousValue}, good to meet you!!”,trigger: “Done”},
```

现在我们有了这么多的逻辑，让我们理解在交互中提供选项是如何工作的。机器人的每个响应都有一个 id。在选项的情况下，我们将有一个数组，而不是消息。

这个数组将由对象组成，这些对象依次具有`value:`、`label:`和`trigger:`

```
{id: “anything next”,options:[{value: “yes”,label: “yes I do”,trigger:”symptom options”},{value: “not at the moment”,label: “not at the moment”,trigger: “Done”}]},
```

基于用户选择的选项，`trigger:`将导致另一个带有`id:`的对象，该对象与用户选择的触发器相匹配。

选项内的函数:如果我们希望直接从聊天机器人执行一个函数，我们可以在用户选择选项时添加一个函数而不是触发器。例如在上面的代码中，我们可以在触发器中添加函数来执行一个函数。

```
{id: “anything next”,options:[{value: “yes”,label: “yes I do”,trigger: () => {
   console.log("yes I do")},{value: “not at the moment”,label: “not at the moment”,trigger: () => {
    console.log("not at the moment") }]},
```

这就是聊天机器人的第一部分和基本要点。使用人工智能来促进对话有许多选择。IBM 也有一个聊天机器人工具，它使得通过聊天机器人获取信息变得非常流畅。

感谢您的阅读。