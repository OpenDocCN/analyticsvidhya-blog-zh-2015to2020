# REDUX 中的规范化不可变状态

> 原文：<https://medium.com/analytics-vidhya/normalized-immutable-state-in-redux-daae0abb2f8?source=collection_archive---------10----------------------->

> 从表面上看，不变状态的概念似乎简单且易于理解。开发人员似乎很难适应它的应用程序…

![](img/be174a4b1de3ec8bd2118fa03a6e0ff7.png)

约书亚·索蒂诺在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

我在理解什么是不可变状态方面没有任何困难，但是当我开始尝试实现它时，我花了一周的时间来尝试应用它。我试图解释为什么一个不可变的状态很难实现，以帮助我自己理解它，也帮助其他人理解它。

为了解释它，我将用三个不同的句子来描述它…

*不可变的状态是一种永远不会被直接修改，而是被改变的值所影响的状态。*

不可变状态是一种被复制、改变然后返回的状态，它的原始内容从不被直接改变。

*不可变状态以其原始形式再现，改变后再返回。*

希望这三句话已经帮助你理解了不变性的概念。对于某些人来说，很难区分可变和不可变状态，因为从概念上看，它们非常相似。一旦这一点被灌输到你的头脑中，一切都归结于实施它。

## 冗余和规范化

规范化是扁平化数据的概念，这带来了巨大的性能提升，因为数据是直接和即时引用的，而不是循环引用。这是因为规范化的嵌套数据是由 id 表示的，这些 id 稍后将用于引用该数据。

在较大的应用程序中，规范化数据的实现变得至关重要，因为需要筛选大量数据来找到特定的文档。一个很好的例子就是在脸书这样的社交媒体平台上的帖子。这些平台中的帖子包含大量具体数据。这些数据可能包括评论、用户信息和帖子信息等。如果这些数据是以 JSON 格式组成的，程序查找嵌套数据会非常耗时，因为这些数据是直接存储的，不会被引用。例如，在一篇文章中有评论，在评论中有每个发帖者的用户 id，有时还有对评论的评论。显然，如果使用规范化数据，通过循环查找嵌套数据是多余的。直接查找数据的剪切速度令人满意，因为客户端可以通过重新加载单个数据来对这些变化做出反应。

**归一化数据**

有几个 NPM 存储库使得规范化过程变得更加容易。一个是“normalizr ”,它获取 JSON 数据并将其转换为规范化数据。

## 已处理的 JSON 数据

```
[ { ** posterId:** "1", ** id:** "1", **title:** "title 1", ** content:** "This is some content.", ** comments:** [
       {
      **  posterId:** "1234", id: "1", content: "This is some comment content"
       }, {
     **   posterId:** "4321", id: "2", content: "This is some comment content" }
    ]}, {
 **  posterId:** "1", ** id:** "2", **title:** "title 2", **content:** "This is some more content.", ** comments:** [
      { **posterId:** "1234", **  id:** "1", **  content:** "This is some more comment content" }, {
      **  posterId:** "4321", **    id:** "2", **  content:** "This is some more comment content" } ]
}
]
```

标准化数据

```
{ **entities:** { **  posts:** { **  1 :** { posterId: "1", title: "title 1", content: "This is some content.", comments: [1, 2], },
     **     2 :** {
               posterId: "1", title: "title 2", content: "This is some content.", comments: [1, 2]
   }}, **   comments:** {
     **    1:** {
             posterId: "1234", content: "This is some comment content" }, **    2:** { posterId: "4321", content: "This is some comment content" }
     }
   }
}
```

好吧，让我们来分解一下。在第一列代码中，有经过处理的 JSON 数据，这是一个包含相应数据集的对象数组。第二个包含一组经过 Normalizr npm 软件包处理的标准化数据。这两组数据的主要区别在于，内容是根据它们的标题分开的。通过使用 Normlizr 的模式建模，用户可以定义如何将他们的数据组织成其规范化的对应物。在这种情况下，定义了“post 模式”和“comments 模式”, comments 模式嵌套在 post 模式中。这样，normlizr 知道当它接收到帖子数据时，它需要提取每个帖子中的所有评论，并将它们转换成自己的对象。然后，它获取每个评论的 id，并将它们放在相应帖子的评论数组中。这样，当一篇文章需要评论的内容时，它只需要将评论数组中的 id 与存储在评论数组中的 id 进行比较。这乍一看似乎很复杂，但从概念上讲是有意义的。这一切都归结于通过查找 id 来处理基于关系的数据，而不是为了找到一个文档而遍历大量数据。

由于在尝试使用规范化数据时，创建和使用 API 可能会非常困难，因此我提供了一些可能有助于这一过程的资源。请花时间通读文档，因为它真正解释了你需要做什么来成功完成你的项目。

## **资源:**

**Redux 的规范化数据文档:**[https://Redux . js . org/recipes/structuring-reducers/normalizing-state-shape](https://redux.js.org/recipes/structuring-reducers/normalizing-state-shape)

**https://www.npmjs.com/package/normalizr**[NPM 套餐](https://www.npmjs.com/package/normalizr)

如果你使用 mongoose，这里有一个 **JSON 规范器插件**用于你的模式:[https://github.com/meanie/mongoose-to-json](https://github.com/meanie/mongoose-to-json)

**有用视频:**

[*数据规格化和冗余:*https://www.youtube.com/watch?v=YvRDgLEY6sE](https://www.youtube.com/watch?v=YvRDgLEY6sE)

*存储数据的最佳方式:【https://www.youtube.com/watch?v=aJxcVidE0I0】T22*