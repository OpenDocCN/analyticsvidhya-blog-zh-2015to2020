# 使用构建器模式简化对象创建

> 原文：<https://medium.com/analytics-vidhya/simplifying-object-creation-with-the-builder-pattern-23ff93eb968a?source=collection_archive---------6----------------------->

![](img/f2ec2caf698c3f7826c49df674bd8e4a.png)

如果你构建它…他们会更容易阅读你的代码。

设计模式是解决软件架构中常见问题的可重用且广为接受的策略。在这篇文章中，我将谈论**构建器模式**。作为[四人组](https://en.wikipedia.org/wiki/Design_Patterns)设计模式之一，构建器模式可以用于封装和控制复杂对象的创建，通过实现一个单独的、具体的构建器类来委托对象创建，而不是试图通过复杂的构造器直接构建。遵循这种模式不仅可以更容易地实例化复杂的类，还可以生成更易于阅读和理解的代码——潜在地防止了错误。

我将展示一个我最近遇到的问题的例子，由于一些要求需要额外的、可选的构造函数参数，我不得不创建一个相当复杂的对象的实例。

## 例子

这里有一个事件类，它有三个必填字段和一些可选字段，用于控制事件的可重试性:

```
**public class** Event {
  **private final** String **type**;
  **private final** String **sender**;
  **private final** Map<String, Object> **payload**;

  **private** Date **created**;
  **private boolean retryable**;
  **private int maxRetries**;
  **private int attemptCount**;
  **private** Date **lastUpdated**;

  **public** Event(String type, String sender, Map<String, Object> payload) {
    **this**.**type** = type;
    **this**.**sender** = sender;
    **this**.**payload** = payload;
    **this**.**created** = **new** Date();
  }

  // additional methods omitted for example

}
```

在这个事件类中，我们需要在构造函数中设置三个 **final** 字段。我们的可选字段在这里帮助我们控制事件的可重试性。

我们可以将它们添加到新的构造函数中，就像这样:

```
**public** Event(String type, String sender, Map<String, Object> payload, **boolean** retryable,
    **int** maxRetries, **int** attemptCount, Date lastUpdated) {
  **this**.**type** = type;
  **this**.**sender** = sender;
  **this**.**payload** = payload;
  **this**.**created** = **new** Date();
  **this**.**retryable** = retryable;
  **this**.**maxRetries** = maxRetries;
  **this**.**attemptCount** = attemptCount;
  **this**.**lastUpdated** = lastUpdated;
}
```

这开始增加复杂性，使我们的代码可读性更差，如果我们需要更多的控制机制来传递给事件，可能会变得混乱。另一个选择是添加一些 setter 方法来设置我们的 Event 类上的可选 retry 字段。

```
**public void** setRetryable(**boolean** retryable) {
  **this**.**retryable** = retryable;
}

**public void** setMaxRetries(**int** maxRetries) {
  **this**.**maxRetries** = maxRetries;
}

**public void** setAttemptCount(**int** attemptCount) {
  **this**.**attemptCount** = attemptCount;
}

**public void** setLastUpdated(Date lastUpdated) {
  **this**.**lastUpdated** = lastUpdated;
}
```

使用 setter 方法，如果我们将来添加更多的字段，我们需要记住为可选字段调用每个 setter 方法。忘记一个可能会在我们的实现中创建一个没有某种空检查的运行时异常。此外，这可能会使事件处于构造函数和 setter 方法类之间的部分状态，这在多线程情况下可能会有进一步的影响。那么，做什么呢？

输入构建器类。这里，我们在事件类中创建了一个嵌套的公共构建器类。我们在我们的事件类中创建它，所以我们不必偏离我们的代码太远，将来，当改变事件对象的接口时，这将允许您或其他开发人员更容易地记住更新构建器。

```
**public class** Event {

  **private final** String **type**;
  **private final** String **sender**;
  **private final** Map<String, Object> **payload**;

  **private** Date **created**;
  **private boolean retryable**;
  **private int maxRetries**;
  **private int attemptCount**;
  **private** Date **lastUpdated**;

  **private** Event(String type, String sender, Map<String, Object> payload) {
    **this**.**type** = type;
    **this**.**sender** = sender;
    **this**.**payload** = payload;
    **this**.**created** = **new** Date();
  }

  **public** Event(String type, String sender, Map<String, Object> payload, **boolean** retryable,
      **int** maxRetries, **int** attemptCount, Date lastUpdated) {
    **this**.**type** = type;
    **this**.**sender** = sender;
    **this**.**payload** = payload;
    **this**.**created** = **new** Date();
    **this**.**retryable** = retryable;
    **this**.**maxRetries** = maxRetries;
    **this**.**attemptCount** = attemptCount;
    **this**.**lastUpdated** = lastUpdated;
  }

  **public void** setRetryable(**boolean** retryable) {
    **this**.**retryable** = retryable;
  }

  **public void** setMaxRetries(**int** maxRetries) {
    **this**.**maxRetries** = maxRetries;
  }

  **public void** setAttemptCount(**int** attemptCount) {
    **this**.**attemptCount** = attemptCount;
  }

  **public void** setLastUpdated(Date lastUpdated) {
    **this**.**lastUpdated** = lastUpdated;
  }

  **public static class** Builder {

    **private final** String **type**;
    **private final** String **sender**;
    **private final** Map<String, Object> **payload**;

    **private boolean retryable** = **false**;
    **private int maxRetries** = 0;
    **private int attemptCount** = 0;
    **private** Date **lastUpdated** = **new** Date();

    **public** Builder(String type, String sender, Map<String, Object> payload) {
      **this**.**type** = type;
      **this**.**sender** = sender;
      **this**.**payload** = payload;
    }

    **public** Builder isRetryable(**boolean** retryable) {
      **this**.**retryable** = retryable;
      **return this**;
    }

    **public** Builder withMaxRetries(**int** maxRetries) {
      **this**.**maxRetries** = maxRetries;
      **return this**;
    }

    **public** Builder withAttemptCount(**int** attemptCount) {
      **this**.**attemptCount** = attemptCount;
      **return this**;
    }

    **public** Builder withLastUpdated(Date lastUpdated) {
      **this**.**lastUpdated** = lastUpdated;
      **return this**;
    }

    **public** Event build() {
      Event event = **new** Event(**type**, **sender**, **payload**);
      event.setRetryable(**retryable**);
      event.setAttemptCount(**attemptCount**);
      event.setMaxRetries(**maxRetries**);
      event.setAttemptCount(**attemptCount**);
      event.setLastUpdated(**lastUpdated**);
      **return** event;
    }
  }

}
```

我们创建的构建器类封装、组装并创建了我们的事件对象，其名称如下:

```
Map<String, Object> payload = **new** HashMap<>();
payload.put(**"message"**, **"Hi!"**);Event event = **new** Event.Builder(**"myType"**, **"Reed"**, payload)
  .isRetryable(**true**)
  .withMaxRetries(2)
  .build();
```

这是一种更干净的方法，而不是创建一个很长的构造函数。

## 委派控制

您可以更进一步，将事件创建的控制权委托给 Builder 类，方法是将事件构造函数设为私有，让 Builder 公共构造函数成为创建新事件对象的唯一接口。

```
**private** Event(String type, String sender, Map<String, Object> payload) {
  **this**.**type** = type;
  **this**.**sender** = sender;
  **this**.**payload** = payload;
  **this**.**created** = **new** Date();
}// disables the ability to call new Event() outside this class
```

## 结论

Builder 模式是四种设计模式中的一种，可用于简化复杂对象的创建，通过消除调用长构造函数或调用多个 setter 方法来创建对象的需要，使您的代码更具可读性，并减少开发人员出错的可能性。

链接:

*   [四人帮设计的图案书](https://en.wikipedia.org/wiki/Design_Patterns)