# Spring 云配置服务器和刷新范围使用的良好实践

> 原文：<https://medium.com/analytics-vidhya/spring-cloud-config-server-and-good-practice-of-refresh-scope-usage-ef65d0fee379?source=collection_archive---------1----------------------->

![](img/04373cc2d0034d64dc290c54852e3e29.png)

奥斯卡·伊尔迪兹在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

十二因素应用程序开发方法强烈建议“**将配置与代码严格分离。”[1]**

Spring Cloud 通过 Spring Cloud Config Server 为这个问题提供了一个解决方案。Spring Cloud 配置服务器定义如下。

“Spring Cloud Config 为分布式系统中的外部化配置提供了服务器端和客户端支持。通过配置服务器，您可以集中管理所有环境中应用程序的外部属性。”[2]

在一个中心位置管理您的配置是一个超级简单而且非常高效的特性。缺乏集中配置会导致经常出错。

另一个好的特性是在改变你的配置后，你可以很容易地从你的应用程序中得到这些，而不需要重启。有很多教程演示了如何在运行时创建配置服务器和刷新属性。我认为**为什么文章**比**如何文章**更重要。所以我没有说明如何创建配置服务器等。我想描述为什么我们在处理运行时属性刷新时使用配置属性。

如果你在网上搜索，有大量的资源只是显示如下控制器的刷新范围。

```
@EnableAutoConfiguration
@ComponentScan
@RestController
@RefreshScope // important!
public class SpringApp {
    @Value("${bar:World!}")
    String bar;

    @RequestMapping("/")
    String hello() {
        return "Hello " + bar + "!";
    }

    public static void main(String[] args) {
        SpringApplication.run(SpringApp.class, args);
    }
}
```

但是在正确的位置选择刷新范围并不容易，如图所示。让我想想下面的场景

```
test:
    value: MyTestValue
```

控制器:

```
@RestController("/test")
public class TestController {

    private final TestService testService;

    public TestController(TestService testService) {
        this.testService = testService;
    }

    @GetMapping("/v1")
    public String test() {
        return testService.getValueWithDelay();
    }

    @GetMapping("/v2")
    public String test2()  {
        return testService.getValue();
    }
}
```

服务:

```
@RefreshScope
@Service
public class TestService {

    @Value("${test.value}")
    private String value;

    public String getValueWithDelay() {
        try {
            Thread.*sleep*(30000L);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return value;
    }

    public String getValue() {
        return value;
    }
}
```

调用[http://127 . 0 . 0 . 1:8080/test/v1](http://127.0.0.1:8080/test/v1)。有 30 秒的时间延迟。30 秒后返回。

调用[http://127 . 0 . 0 . 1:8080/test/v](http://127.0.0.1:8080/test/v1)2。它在 6 毫秒后返回。

似乎一切都很完美。我们换个场景吧。

*   **调用**[**http://127 . 0 . 0 . 1:8080/test/v1**](http://127.0.0.1:8080/test/v1)
*   **将测试值更改为 MyTestValueChanged**
*   **刷新属性 localhost:8080/actuator/Refresh**
*   **调用**[**http://127 . 0 . 0 . 1:8080/test/v**](http://127.0.0.1:8080/test/v1)**2**

**V2 端点大约在 10-20 秒后返回。(你需要在 v1 执行完成之前完成这些操作)**

我们在服务类中使用了刷新范围。如果值在服务类中使用，那么其他方法将等待，直到这个值被上下文替换。在我们现代的软件开发环境中，甚至毫秒都很重要。V1 和 V2 是不同的操作，但由于刷新范围的使用，它们会相互阻塞。更糟糕的是，跟踪哪个值将影响哪个服务变得非常困难。

在这一点上，配置属性的概念变得复杂起来。如果在配置属性中使用刷新范围而不是服务，那么可以解决不同服务相互阻塞的问题。因为上下文只刷新配置属性 bean。并且刷新速度非常快。这就是为什么你应该使用配置属性并在其中使用刷新范围。

```
@RefreshScope
@Configuration
@ConfigurationProperties(prefix = "test")
public class TestConfiguration {

    private String value;

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }
}
```

服务:

```
@Service
public class TestService {
    private final TestConfiguration testConfiguration;

    public TestService(TestConfiguration testConfiguration) {
        this.testConfiguration = testConfiguration;
    }

    public String getValueWithDelay() {
        try {
            Thread.*sleep*(30000L);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return testConfiguration.getValue();
    }

    public String getValue() {
        return testConfiguration.getValue();
    }
}
```

*   致电[http://127 . 0 . 0 . 1:8080/test/v1](http://127.0.0.1:8080/test/v1)
*   将测试值更改为 MyTestValueChangedAgain
*   刷新属性本地主机:8080/actuator/refresh
*   致电[http://127 . 0 . 0 . 1:8080/test/v](http://127.0.0.1:8080/test/v1)2

V2 端点大约在 10 毫秒后返回。没有障碍。

问题解决了。

PS:特别感谢 Ercan Sormaz。

参考资料:

1.  [https://12factor.net/config](https://12factor.net/config)
2.  [https://cloud.spring.io/spring-cloud-config/reference/html/](https://cloud.spring.io/spring-cloud-config/reference/html/)