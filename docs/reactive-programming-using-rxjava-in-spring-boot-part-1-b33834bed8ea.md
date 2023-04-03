# 在 spring-boot 中使用 RxJava 进行反应式编程:第 1 部分

> 原文：<https://medium.com/analytics-vidhya/reactive-programming-using-rxjava-in-spring-boot-part-1-b33834bed8ea?source=collection_archive---------4----------------------->

![](img/9ea475700b4a6e164a661c3ca1e32181.png)

欢迎朋友，很高兴见到你。读完这篇文章后，我可以确定你可以马上开始写代码。

有了 reactive，你就有了数据流，你必须观察它们，并在值发出时做出反应。

因此，您需要传感器数据、HTTP 请求、通知、流数据、连续位置等数据流，这些数据可以异步处理。

大多数 web 应用程序都是命令式的，操作本质上是顺序的，你有一个调用堆栈，一个操作被另一个操作阻塞，直到它完成，然后只有另一个操作执行。

在**事件驱动应用**的情况下，每个事件单独运行。基本上它保持并发性。这种方法的问题是，我们将有大量的回调。在这里，反应式编程开始发挥作用。

如果你想了解更多关于反应式编程的知识，你可以点击这个链接[https://dzone . com/articles/5-things-to-know-about-reactive-programming](https://dzone.com/articles/5-things-to-know-about-reactive-programming)

现在我们可以直接进入 Rxjava 库

如果您正在使用 Gradle、maven 或任何其他构建系统，您可以检查此链接，并在您的构建系统中包含您需要的所有依赖项

[](https://github.com/ReactiveX/RxJava/wiki/Getting-Started) [## react vex/rx Java

### 这是一个库，用于使用…

github.com](https://github.com/ReactiveX/RxJava/wiki/Getting-Started) 

可观察接口是 Rxjava 库的基本组件

可观察的接口传递三种类型的事件。

1.  **onNext** :传递数据(发射)给观察者实例。
2.  onComplete :当观察者完成所有发射时调用
3.  **onError** :发生异常或错误时调用。

这就是我们如何创造可观察的

```
 Observable<Object> source= Observable.create(emmiter->{
          try{

               emmiter.onNext("first");
               emmiter.onNext("second");
               emmiter.onNext("third");
               emmiter.onComplete(); }catch(Exception ex){
               emmiter.onError(ex);
            }
           }); }
```

在 onNext 函数中，我们发出三个字符串值，一旦发出完成，我们就调用 onComplete 来通知发出完成。

现在我们有了数据流，我们必须观察它。为此，我们需要 Observer 类来订阅 Observable 类并实现所有三个功能。

```
Observer<String> observer = new Observer<String>(){ @Override
 public void onComplete() {
  System.out.println("Emission completed ..");
 }

 @Override
 public void onError(Throwable arg0) {
  System.out.println("Error occurred: "+arg0.getLocalizedMessage());
 } @Override
 public void onNext(String arg0) {
  System.out.println("Data is .. "+arg0);
 }
};
```

现在我们需要给观察者订阅可观察的，我们可以通过

```
source.subscribe(observer);
```

当我们运行上面的代码时，我们将得到输出。

```
Data is .. first
Data is .. second
Data is .. third
Emission completed ..
```

还有其他一些方法来创造可观的，如

```
Observable<Object> source= Observable.just("first", "second", "third", "four");
```

这里我们使用 just 函数来传递排放数据。

```
List<String> countryList = Arrays.asList("first", "second", "third");Observable<String> source = Observable.fromIterable(countryList);
```

从列表中，我们也可以创建可观的。

**可观测量中的一些基本算子**

1.  **映射和过滤**

Map 操作符应用于源可观察对象发出的每个项目，并返回发出这些函数应用结果的可观察对象。

```
Observable<String> source= Observable.just("map operator ","source observer returns in upper case","using map we are returning in upper case","using map operator"); 
        source.map(String::toUpperCase).subscribe(System.out::println);
```

过滤器-根据条件过滤可观测的发射。它只返回那些符合条件的排放

```
Observable<Integer> source = Observable.range(0,10); source.filter(i-> i%2==0).map(i-> i*i).subscribe(System.out::println);
```

**2。开始、重复、缩小、扫描**

`startWith()`将给定元素附加到发射的开始

```
Observable<String> source = Observable.just("Rx", "Java", " Operators", " Tutorial");

source.startWith("We are discussing here about ").subscribe(System.out::print);
```

`reduce()`它将下一个值添加到先前添加的值中。

```
Observable<Integer> source = Observable.range(1, 5);source.reduce((integer, integer2) -> integer + integer2).subscribe(System.out::println);//Output
1
3
and so on. 
```

repeat()重复发射两次

```
Observable.just("Java", "Nodejs", "Python")                 .repeat(2).subscribe(System.out::println);//Output
Java
Nodejs
Python
Java
Nodejs
Python
```

**3。排序**对元素进行排序

```
Observable<String> source = Observable.just(1,3,4,2,8,6,7)                 source.sorted.subscribe(System.out::println);
```

**合并相关运算符**

当我们需要合并两个可观测的。我们需要对第一个可观测值的发射进行一些操作，并返回一个新的可观测值。

假设一个网络调用以编码的形式返回数据流，现在你解码它，然后返回一个新的可观察值，然后一些用户可以使用这些数据。

一些算子用来合并多个可观测量。

1.  **平面图**

平面图将一个可观测物发出的项目转换成新的可观测物。

```
Observable<String> source1= Observable.just("abcdef", "123456", "ABCDEF");source1.flatMap(emission-> Observable.fromArray(emission.split(""))).subscribe(System.out::println);// Outputa
b
c
d
e
f
...
```

Second observable 逐个获取元素，然后打印该元素的每个字符

2.**串联图**

它类似于平面地图，只是后者不维持秩序。因此，如果您需要在发射中排序，请使用 concatMap。

> **一个使用 Rxjava 的样例用例项目**

**您可以在**中找到完整的项目代码

[](https://github.com/kshivam213/demo_rxjava/tree/master/src/main/java/com/rxjava/news) [## kshivam213/demo_rxjava

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/kshivam213/demo_rxjava/tree/master/src/main/java/com/rxjava/news) 

**要求**

1.  与 https://newsapi.org/[整合](https://newsapi.org/)
2.  每天我们必须获取最新的新闻，并使用 cronJob 存储在数据库中。这里我们要用 Rxjava 调用 API，并存储在 DB 中。
3.  公开根据国家、语言和类别等条件返回最新新闻的端点。

更多细节请参考上面 GitHub 链接上的项目代码。

**包装**

我希望你喜欢这个教程，在下一部分，我将讨论可流动，反压力，反压力策略，分片，重试机制和缓存。

**有用资源**

1.  [https://www.baeldung.com/rxjava-tutorial](https://www.baeldung.com/rxjava-tutorial)