# Apache Beam:在一个没有 Spring/Guice 的世界中管理您的依赖项…

> 原文：<https://medium.com/analytics-vidhya/apache-beam-a-minor-tip-for-cleaner-code-f293df3e3372?source=collection_archive---------4----------------------->

![](img/e31163f752dec524e02bfa9b3478d057.png)

布莱克·康纳利在 [Unsplash](https://unsplash.com/s/photos/code-review?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

这不是一个好主意，只是 ApacheBeam 中隐藏的特性，你可能会在阅读整个文档时忽略它…

Apache Beam 是一个健壮的分布式流数据管道框架，它是 SDK，有许多运行程序:Flink、DataFlow 等等。

当我深入管道并必须与外部服务(如 RedisClient、DB 连接池或一些内部配置服务)进行通信时，我感觉我的代码开始变得混乱。

我怀念 Spring 或者其他 DI 框架。我在一个地方定义和初始化我的服务，在另一个地方在需要时注入它们，如果没有这种能力，我必须将创建服务所需的所有参数和依赖项传递到我的`DoFn`中。
除此之外，通常，服务应该是 **singletons** ，这在高规模应用中非常重要，想象一下，当我的服务包含一个数据库连接池，而它不是 singleton 时，在这种情况下，我可能会很快达到数据库中允许的最大连接数。

让我们用一个代码示例(丑陋的一个)来总结一下缺少了什么:

1.  Singleton:创建被锁阻塞。
2.  耦合:一个函数必须知道所有的 Redis 创建属性，即使它只需要使用它。
3.  硬重构:如果我现在需要向 Redis 创建添加更多参数，我必须将它传递给构造函数，如果从许多地方调用它，我必须修改它们，测试可能会失败，并且这种重构容易出错。
4.  测试是困难的:许多静态的、复杂的模仿依赖。

解决方案是:

我们可能有包含所有 Redis 配置的管道选项:

我们只需添加一个特殊选项“`*RedissonClient* getRedissonClient();`”，并用`@Default.InstanceFactory(RedisClientFactory.*class*)`对其进行注释。注释中的类是一个“知道”如何初始化我的目标服务的工厂，每个工人只需创建一次。

这里是工厂:

上面的工厂获得了创建 Redis 客户端所需的所有属性，依赖项/参数的更改或添加仅在这里生效。

变化后的`DoFn`:

**注意事项:**

1.  在`@ProcessElement`范围外没有 DI，只能从`DoFn`功能访问选项。
2.  毫无特色，没有 AOP，实例生命周期挂钩。
3.  实例总是在懒惰模式下初始化，只有当你请求它们时，它们才会被创建，没有办法克服这一点。

为什么不是春天？

Apache Beam 没有在 worker 初始化时暴露 hook，所以我不能使用 Spring，因为它需要每个 worker 上的一些起始点来初始化它的上下文和 bean。

*是的，一个小的改变，非常直接，但是如果你不及时遵循它，你可能会得到耦合的、不可测试的代码，并且难以维护。*