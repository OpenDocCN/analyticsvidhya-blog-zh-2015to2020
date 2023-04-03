# 模型是 Django 的根..！

> 原文：<https://medium.com/analytics-vidhya/models-are-root-of-django-a0c16ea5673a?source=collection_archive---------13----------------------->

![](img/68e6d58b6f87f42520d3ae4dbb9fa1b5.png)

来源:网络征稿

在薄弱的基础上，你不可能建造伟大的建筑。戈登·b·欣克利

模型是 ORM 的基础。使用 Django，您可以将效率提高 10 倍！就复杂性和及时性而言。但是首先，您必须了解 Django 是如何处理数据库的！

在写模型的时候，我们经常犯错误，因为我们不知道做同一件事情的不同方法，可能是正确的方法。有时，由于糟糕的模型设计，也会导致性能障碍。解决这种障碍是很困难的，因为它需要结构上的改变，我们最终要做大量的迁移来建立和运行东西。

在本文中，我将分享我在 Django 中创建模型的一些见解。

## 为正确的事情建立正确的关系！

*当第一个表中的一条记录只对应相关表中的一条记录时，使用一对一字段关系*。例如，餐馆只有一个地方供应膳食。对不对…？

而且餐厅不止一个服务员。在这种情况下，我们使用 ForeignKey，这是一对多关系的例子！

*当一个表中的多条记录与另一个表中的多条记录相关联时，就会出现多对多关系*。例如，一个比萨饼可以有许多配料。许多披萨上都有浇头。这是使用`ManyToManyField`的好地方。

## 索引模型！

Django 将在任何外键上自动创建一个 B 树索引。它们将在 *where 子句*中使用。

```
Asset.objects.filter(id=1)  # filtering on primary key properties
Asset.objects.filter(project__id=1)  # filtering on foreign key properties with where clause
```

通常用于搜索和显示的模型，在这种情况下，我们可以使用索引。例如，产品名称(CharField)字段需要对其名称字段进行索引。

这个例子就是 `models.TextField(db_index = True)`

这将用于**对列属性的过滤:** `Project.objects.filter(name__contains="Foo")`

对外键属性进行过滤的**也是如此:**

`Asset.objects.filter(project__name__contains="Foo")`

一个好的规则是，如果你要排序或过滤，它应该被索引。然而，永远记住，在数据库中使用索引是有代价的，所以要明智地使用它。

## UniqueConstraint 和 CheckConstraint

这两个约束用于模型级别的验证，在模型级别进行验证是最佳实践之一。

让我们看一个例子，`**UniqueConstraint(fields=['room', 'date'])**`它确保每个房间在每个日期只能被预订一次；这个解决方案很贴切！

CheckConstraint 顾名思义，它会在保存之前检查值。

## 预取相关和选择相关

`prefetch_related`用于从*多对多* & *多对一*关系中预取数据，并从*一对一*关系等单值关系中选择数据。我们有`select_related.`

例如，您构建了一个模型和一个与其他模型有关系的模型。当有请求时，您也会收到对其相关数据的查询。姜戈有优秀的👌从关系中访问数据的机制，如`**book.author.name**`

但是如果不使用`prefetchd_related`和`selected_related,`，Django 会为每个关系数据创建一个对数据库的请求。这会导致性能障碍😞。为了克服这些困难，我们有了`prefetchd_related`和`selected_related` 的概念😎。

## 经验法则👍

1.  总是在模型级别验证字段
2.  查询不应该在循环中
3.  了解查询集的工作原理
4.  恰当的做法是制作模型方法。这些模型实例将可以访问它所有方法。

例子:`get_absolute_url(), __str__(), publish(), get_cost() get_info() calculate_total()`

仅此而已。你认为会有其他可能的解决方案吗？请在下面的评论中告诉我。

> Django 是一个很好的网络框架，但是我认为，让它成为一个 ORM 人的好框架😍

# 结论

我的结论很简单！没有一个坚实的基础，任何有价值的东西你都会遇到困难。模型是钥匙的基础。首先，聪明地工作，最终把它做好。

如果你喜欢这篇文章，那就鼓掌吧👏…!

回头见，继续编码，玩得开心❤
贾米勒。