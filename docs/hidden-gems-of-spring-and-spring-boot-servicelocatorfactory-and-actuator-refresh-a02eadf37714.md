# 弹簧和 Spring Boot 的隐藏宝石—服务定位工厂和/执行器/更新

> 原文：<https://medium.com/analytics-vidhya/hidden-gems-of-spring-and-spring-boot-servicelocatorfactory-and-actuator-refresh-a02eadf37714?source=collection_archive---------0----------------------->

**问题陈述**

我设计的 API 基本上调用各种第三方 API，如订餐、预订火车票、寻找免费停车场，并将响应返回给前端。通常情况下，供应商会不断变化。例如，对于订单食品，第三方供应商可以根据变更的合同从 Zomato 切换到 Swiggy。(请注意，这些名称仅用作示例)。现在，对于点餐流程/任何其他流程的要求是

1.  食品服务 API 应该作为一个接口，隐藏供应商的转换。
2.  在食品服务供应商属性从 Zomato 配置更改为 Swiggy 配置的情况下，应该有一个运行时更改，而不需要重启应用程序。

对于第一个需求，服务工厂设计模式是最适合的。为了满足第二个需求，spring cloud config server 是最合适的。

下面是 Spring 的服务工厂模式通过配置在不同的服务实现之间切换的步骤

步骤 1 将服务定位器工厂 bean 添加到 spring boot 配置中

```
@Beanpublic ServiceLocatorFactoryBean slfbForServiceFactory() { ServiceLocatorFactoryBean bean = new ServiceLocatorFactoryBean(); bean.setServiceLocatorInterface(ServiceFactory.class); return bean;}
```

步骤 2 创建一个接口 FoodService 以及 FoodService 的两个服务实现，如下所示。

```
public interface FoodService {List<StoreLocation> findStores(LatLong location, double radiusInMeters) ; } @Service("ZomatoFoodService")
public class ZomatoFoodService implements FoodService {
} @Service("SwiggyFoodService")
public class ZomatoFoodService implements FoodService {
}
```

步骤 3 创建 ServiceFactory。基本上，这是一个基于限定符给出服务实现实例的接口。

```
public interface ServiceFactory { FoodService getFoodService(String qualifier); }
```

步骤 4 创建配置文件来传递服务实现的限定符名称。

```
@Configuration@ConfigurationProperties(prefix = "service")@Data*#lombok annotation*public class ServiceConfig {
 String foodVendor; }
```

步骤 5 在 application.properties 文件中添加属性及其值，如下所示。

```
service.foodVendor = ZomatoFoodService
```

通过以上步骤，我们已经使服务实现注入可配置。

下面是你如何从控制器/任何你需要使用服务实现的类中引用你的食物服务。

```
@RestControllerpublic class FoodController { @Autowiredprivate ServiceConfig serviceConfig; @Autowiredprivate ServiceFactory serviceFactory; private FoodService getFoodService() { return serviceFactory.getFoodService(serviceConfig.getFoodVendor()); }
```

通过以上步骤，我们已经让服务实现切换属性文件的职责。现在剩下的就是将应用程序的属性外部化，然后在不重启应用程序的情况下刷新它们。这项工作是由 spring 配置服务器完成的。

Spring Cloud Config Server——Spring Cloud Config Server 为外部配置(名称-值对或等效的 YAML 内容)提供了一个基于 HTTP 资源的 API。通过使用@EnableConfigServer 注释，服务器可嵌入到 Spring Boot 应用程序中。(这就是 spring 文档所陈述的)——我没有使用这个注释，而是使用了 actuator 来刷新。

现在有两种方法可以实现 spring config server——嵌入到微服务/独立服务中，为所有微服务提供配置。第二个可以作为单点故障，我们选择了第一个。

Spring Cloud 嵌入式配置服务器和/执行器/刷新

第一步在你的 gradle 文件中添加 spring cloud 依赖，如下所示。

```
ext {
   set('springCloudVersion', 'Finchley.SR2')
}dependencyManagement {
   imports {
      mavenBom "org.springframework.cloud:spring-cloud-dependencies:${springCloudVersion}"
   } dependencies {implementation 'org.springframework.cloud:spring-cloud-config-server'}
}
```

步骤 2:创建 bootstrap.yml 并添加以下属性。

如果您想要直接从后端存储库(而不是从配置服务器)读取应用程序的配置，那么您基本上想要一个没有端点的嵌入式配置服务器。您可以通过不使用@EnableConfigServer 批注来完全关闭端点(set spring . cloud . config . server . bootstrap = true)。

请注意，我们没有使用以下属性，因为我们没有独立的配置服务器。

```
spring.cloud.config.urispring:
*# change the application name for your project name*
  application:
    name: foodorder-api--- spring:
  cloud:
    config:
      server:
        bootstrap: true
        git:
          uri: *#ssh url of your git repo* searchPaths: foodorder-api
          username: *# put the username of git repo*
          password: *# put the password of git repo*
```

将在 Spring boot 版本中添加属性，以启用除/health 和/info 之外的所有执行器端点。

```
*#property to expose /actuator/refresh endpoint to expose embedded config server refresh*
management.endpoints.web.exposure.include=refresh
*#property to expose /actuator other end points disabled by default*
management.endpoints.web.exposure.include=*
```

您可以通过向应用程序的刷新端点发送一个空的 HTTP POST 来调用刷新执行器端点，[HTTP://localhost:8080/Actuator/refresh](http://localhost:8080/actuator/refresh)

```
$ curl localhost:8080/actuator/refresh -d {} -H "Content-Type: application/json"
```

提示-

1.  /refresh endpoint 仅刷新那些用@ConfigurationProperties 批注的属性。它不像@Value 那样刷新那些属性。对于@Value，您需要在使用它的类中包含@RefreshScope。
2.  在 pivotal cloud foundry 上，现在配置服务器作为一项服务提供。
3.  对于 docker 部署，确保可以从 docker swarm 访问 git。请确保适当的证书也存在，因为我们使用 ssh 连接进行 git 结帐。

参考资料:-

1.  [https://cloud . spring . io/spring-cloud-config/multi/multi _ _ spring _ cloud _ config _ server . html](https://cloud.spring.io/spring-cloud-config/multi/multi__spring_cloud_config_server.html)
2.  【https://spring.io/guides/gs/centralized-configuration/ 