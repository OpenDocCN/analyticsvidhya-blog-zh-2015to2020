# 贝叶斯定理

> 原文：<https://medium.com/analytics-vidhya/bayes-theorem-d8904bb76367?source=collection_archive---------14----------------------->

![](img/b582bda13b472d1e97b54dac7e026ca8.png)

P **概率*无非是事件发生的可能性***

***事件*** →它是一个实验的结果或我们经历的结果

例如，假设您正在掷骰子

实验→滚动六面模具

事件→掷骰子后的结果是 4

```
Possible outcomes after rolling the die = [1,2,3,4,5,6]Total number of outcomes = 6
```

结果为 4 的可能性是事件发生的概率= 4

```
P(Event = 4) = (no_of_outcomes = 4)/(total_no_outcomes) = 1/6
```

类似地考虑上述实验

```
P(Event = 1) = 1/6P(Event = 2) = 1/6P(Event = 3) = 1/6P(Event = 5) = 1/6P(Event = 6) = 1/6
```

> p(事件)总是介于 0 和 1 之间，因为一个事件在实验中发生的次数总是小于可能结果的总数

> ***所有可能事件的概率之和等于一***

考虑上面的实验

```
P(Event = 1) + P(Event = 2) + P(Event = 3) +
P(Event = 4) + P(Event = 5) + P(Event = 6) = 1
```

## ***例题***

10 人去了商店，其中 4 人购买了耐克产品，3 人购买了彪马产品，1 人购买了阿迪达斯产品，其余 2 人没有购买任何产品

以下是可能的事件

```
1\. Person buying Nike product2\. Person buying Adidas product3\. Person buying Puma product4\. Person not buying any product 
```

所以概率

```
P(Event = Nike) = 4/10P(Event = Puma) = 3/10P(Event = Adidas) = 1/10
```

# 条件概率

> **条件概率**就是一个事件发生的概率，假设另一个事件已经发生。

```
P(A|B) = Prob of event A occuring based on the occuring of event BP(A & B) = Prob of both events occuring P(A) = No of times event A occurs / total eventsP(B) = No of times event B occurs / total eventsP(A & B) = No of times both events occur / total eventsP(A|B) = No of times both A and B occurs / No of times B occursP(A|B) = P(A & B)/P(B)
```

## 例子

考虑 100 个人去购物。

其中 40 人是欧洲人，60 人是非欧洲人。

其中 70 人购买了耐克产品。

20 名欧洲人购买了耐克产品，其余的更喜欢阿迪达斯

可能的事件

```
1\. Person being an EU2\. Person being a non-EU3\. Person buy Nike related product4\. Person not buying Nike related product
```

可能性

```
P(EU) = 40/100 = 0.4P(non-EU) = 60/100 = 0.6P(Nike) = 70/100 = 0.7P(non-Nike) = 30/100 = 0.3
```

20 名欧洲人购买了耐克产品

```
P(EU & Nike) = Prob of a person being from Europe and bought a Nike ProductP(EU & Nike) = 20/100 = 0.2
```

假设某人购买了耐克的产品，那么这个人来自欧洲的概率有多大

```
P(EU|Nike) = P(EU) among persons who bought NikeP(EU|Nike) = P(EU & Nike)/P(Nike) = 0.2/0.7 = 2/7 = 0.28
```

> 因此，假设一个人从耐克购买了一件产品，那么这个人来自欧洲的概率是 0.28

# 贝叶斯定理

假设我们想找出条件概率 P(A|B ),但是我们已经根据 P(B|A)知道了它。

```
P(A|B) in terms of P(A), P(B) and P(B|A)P(A|B) = (P(B|A) * P(A))/((P(B|A) * P(A)) + (P(B|not-A) * P(not-A)))
```

考虑同样的例子

```
P(EU|Nike) = 0.28P(EU) = 0.4P(Nike) = 0.7P(not-Nike) = 1 - P(Nike) = 1 - 0.7 = 0.3
```

使用条件概率，假设一个人没有购买耐克产品，这个人来自欧洲的概率

```
P(EU|not-Nike) = P(EU & not-Nike)/P(not-Nike)P(EU|not-Nike) = (20/100)/(0.3) = 2/3
```

现在，假设一个人是欧洲人，用 Baye 定理来衡量这个人购买耐克产品的概率

```
P(Nike|EU) = (P(EU|Nike) * P(Nike)) / ((P(EU|Nike)*P(Nike)) + (P(EU|not-Nike) * P(not-Nike)))P(EU|Nike) * P(Nike) = 2/7 * 7/10 = 0.2P(EU|not-Nike) * P(not-Nike) = 2/3 * 3/10 = 0.2P(Nike|EU) = (2/7 * 7/10)/((2/7 * 7/10) + (2/3 * 3/10)) = 0.2/0.4P(Nike|EU) = 0.5
```

> 因此，假设一个人来自欧洲，他购买耐克产品的概率是 0.5

现在测量一个人购买耐克产品的概率，假设这个人属于欧洲，不使用贝耶定理

```
P(Nike|EU) = P(Nike & EU) / P(EU)P(Nike & EU) = 20/100 = 2/10P(EU) = 40/100 = 4/10P(Nike|EU) = (2/10)/(4/10) = 0.5
```

> 注意， **P(耐克|欧盟)和 P(欧盟|耐克)**是不一样的。条件概率不只是建立两个事件之间的关系，它是基于两个事件的排序的度量