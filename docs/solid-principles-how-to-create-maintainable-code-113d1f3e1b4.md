# 坚实的原则:如何创建可维护的代码

> 原文：<https://medium.com/analytics-vidhya/solid-principles-how-to-create-maintainable-code-113d1f3e1b4?source=collection_archive---------20----------------------->

![](img/39cee5f8dac52307cbe96e0244a9e48a.png)

在之前的帖子中，我们讨论了编写代码时应该避免的原则。这里可以看[。在这篇文章中，我们将着眼于坚实的设计原则，编程标准，所有开发人员必须很好地理解，以创建良好的架构。](/@marat_badykov/stupid-principles-what-you-should-avoid-when-writing-code-2d84eb7818fd)

SOLID 是以下单词的首字母缩略词:

*   单一责任
*   开-关
*   利斯科夫替代
*   界面分离
*   依赖性倒置

# 单一责任

*   每个对象应该被分配一个单独的责任。
*   特定的类应该解决特定的问题。

## 愚蠢的例子

```
**class** **Order**
{
    **public** **function** **calculateTotalSum**() {**...**}
    **public** **function** **getItems**() {**...**}
    **public** **function** **getItemCount**() {**...**}
    **public** **function** **addItem**() {**...**}
    **public** **function** **deleteItem**() {**...**}

    **public** **function** **load**() {**...**}
    **public** **function** **save**() {**...**}
    **public** **function** **update**() {**...**}
    **public** **function** **delete**() {**...**}

    **public** **function** **printOrder**() {**...**}
    **public** **function** **showOrder**() {**...**}
}
```

## 实体示例

```
**class** **Order**
{
    **public** **function** **calculateTotalSum**() {**...**}
    **public** **function** **getItems**() {**...**}
    **public** **function** **getItemCount**() {**...**}
    **public** **function** **addItem**() {**...**}
    **public** **function** **deleteItem**() {**...**}
}

**class** **OrderRepository**
{
    **public** **function** **load**() {**...**}
    **public** **function** **save**() {**...**}
    **public** **function** **update**() {**...**}
    **public** **function** **delete**() {**...**}
}

**class** **OrderViewer**
{
    **public** **function** **printOrder**() {**...**}
    **public** **function** **showOrder**() {**...**}
}
```

# 开-关

*   软件实体必须对扩展开放，但对修改关闭。
*   所有的类、函数等。应该这样设计，以便改变它们的行为，不需要改变它们的源代码。

## 愚蠢的例子

```
**class** **OrderRepository**
{
    **public** **function** **load**($orderId)
    {
        $pdo **=** **new** PDO($this**->**config**->**getDsn(), $this**->**config**->**getDBUser(), $this**->**config**->**getDBPassword());
        $statement **=** $pdo**->**prepare('SELECT * FROM `orders` WHERE id=:id');
        $statement**->**execute(**array**(':id' **=>** $orderId));
        **return** $query**->**fetchObject('Order');
    }
    **public** **function** **save**($order) {**...**}
    **public** **function** **update**($order) {**...**}
    **public** **function** **delete**($order) {**...**}
}
```

## 实体示例

```
{
    **private** $source;

    **public** **function** **setSource**(IOrderSource $source)
    {
        $this**->**source **=** $source;
    }

    **public** **function** **load**($orderId)
    {
        **return** $this**->**source**->**load($orderId);
    }

    **public** **function** **save**($order)
    {
        **return** $this**->**source**->**save($order);
    }

    **public** **function** **update**($order) {**...**};
    **public** **function** **delete**($order) {**...**};
}

**interface** **IOrderSource**
{
    **public** **function** **load**($orderId);
    **public** **function** **save**($order);
    **public** **function** **update**($order);
    **public** **function** **delete**($order);
}

**class** **MySQLOrderSource** **implements** IOrderSource
{
    **public** **function** **load**($orderId) {**...**};
    **public** **function** **save**($order) {**...**}
    **public** **function** **update**($order) {**...**}
    **public** **function** **delete**($order) {**...**}
}

**class** **ApiOrderSource** **implements** IOrderSource
{
    **public** **function** **load**($orderId) {**...**};
    **public** **function** **save**($order) {**...**}
    **public** **function** **update**($order) {**...**}
    **public** **function** **delete**($order) {**...**}
}
```

# 利斯科夫替代

*   程序中的对象可以被它们的继承者替换，而不改变程序的属性。
*   使用类继承时，代码执行的结果应该是可预测的，并且不会更改方法的属性。
*   应该可以替换基本类型的任何子类型。
*   使用基类引用的函数应该能够在不知道的情况下使用派生类的对象。

## 愚蠢的例子

```
**class** **Rectangle**
{
    **protected** $width;
    **protected** $height;

    **public** setWidth($width)
    {
        $this**->**width **=** $width;
    }

    **public** setHeight($height)
    {
        $this**->**height **=** $height;
    }

    **public** **function** **getWidth**()
    {
        **return** $this**->**width;
    }

    **public** **function** **getHeight**()
    {
        **return** $this**->**height;
    }
}

**class** **Square** **extends** Rectangle
{
    **public** setWidth($width)
    {
        **parent::**setWidth($width);
        **parent::**setHeight($width);
    }

    **public** setHeight($height)
    {
        **parent::**setHeight($height);
        **parent::**setWidth($height);
    }
}

**function** **calculateRectangleSquare**(Rectangle $rectangle, $width, $height)
{
    $rectangle**->**setWidth($width);
    $rectangle**->**setHeight($height);
    **return** $rectangle**->**getHeight ***** $rectangle**->**getWidth;
}

calculateRectangleSquare(**new** Rectangle, 4, 5); *// 20*
calculateRectangleSquare(**new** Square, 4, 5); *// 25 ???*
```

## 实体示例

```
**class** **Rectangle**
{
    **protected** $width;
    **protected** $height;

    **public** setWidth($width)
    {
        $this**->**width **=** $width;
    }

    **public** setHeight($height)
    {
        $this**->**height **=** $height;
    }

    **public** **function** **getWidth**()
    {
        **return** $this**->**width;
    }

    **public** **function** **getHeight**()
    {
        **return** $this**->**height;
    }
}

**class** **Square**
{
    **protected** $size;

    **public** setSize($size)
    {
        $this**->**size **=** $size;
    }

    **public** **function** **getSize**()
    {
        **return** $this**->**size;
    }
}
```

# 界面分离

*   许多专用接口比一个通用接口要好。
*   遵循这一原则是必要的，这样使用/实现接口的客户端类只知道它们使用的方法，从而减少未使用的代码量。

## 愚蠢的例子

```
**interface** **IItem**
{
    **public** **function** **applyDiscount**($discount);
    **public** **function** **applyPromocode**($promocode);

    **public** **function** **setColor**($color);
    **public** **function** **setSize**($size);

    **public** **function** **setCondition**($condition);
    **public** **function** **setPrice**($price);
}
```

## 实体示例

```
**interface** **IItem**
{
    **public** **function** **setCondition**($condition);
    **public** **function** **setPrice**($price);
}

**interface** **IClothes**
{
    **public** **function** **setColor**($color);
    **public** **function** **setSize**($size);
    **public** **function** **setMaterial**($material);
}

**interface** **IDiscountable**
{
    **public** **function** **applyDiscount**($discount);
    **public** **function** **applyPromocode**($promocode);
}
```

# 依赖性倒置

*   系统中的依赖关系是基于抽象的。
*   顶层模块独立于底层模块。
*   抽象不应该依赖于细节。
*   细节应该依赖于抽象。

## 愚蠢的例子

```
**class** **Customer**
{
    **private** $currentOrder **=** **null**;

    **public** **function** **buyItems**()
    {
        **if** (is_null($this**->**currentOrder)) {
            **return** **false**;
        }

        $processor **=** **new** OrderProcessor(); *// !!!*
        **return** $processor**->**checkout($this**->**currentOrder);
    }

    **public** **function** **addItem**($item)
    {
        **if** (is_null($this**->**currentOrder)) {
            $this**->**currentOrder **=** **new** Order();
        }

        **return** $this**->**currentOrder**->**addItem($item);
    }

    **public** **function** **deleteItem**($item)
    {
        **if** (is_null($this**->**currentOrder)) {
            **return** **false**;
        }

        **return** $this**->**currentOrder**->**deleteItem($item);
    }
}

**class** **OrderProcessor**
{
    **public** **function** **checkout**($order) {**...**}
}
```

## 实体示例

```
**class** **Customer**
{
    **private** $currentOrder **=** **null**;

    **public** **function** **buyItems**(IOrderProcessor $processor)
    {
        **if** (is_null($this**->**currentOrder)) {
            **return** **false**;
        }

        **return** $processor**->**checkout($this**->**currentOrder);
    }

    **public** **function** **addItem**($item){
        **if** (is_null($this**->**currentOrder)) {
            $this**->**currentOrder **=** **new** Order();
        }

        **return** $this**->**currentOrder**->**addItem($item);
    }

    **public** **function** **deleteItem**($item) {
        **if** (is_null($this**->**currentOrder)) {
            **return** **false**;
        }

        **return** $this**->**currentOrder**->**deleteItem($item);
    }
}

**interface** **IOrderProcessor**
{
    **public** **function** **checkout**($order);
}

**class** **OrderProcessor** **implements** IOrderProcessor
{
    **public** **function** **checkout**($order) {**...**}
}
```

# 摘要

## 单一责任原则

> *应该给每个对象分配一个单独的责任。“为了做到这一点，我们检查我们有多少理由改变这个类——如果不止一个，那么这个类应该被打破。*

## 开放/封闭的原则(开放-封闭)

> *软件实体必须对扩展开放，但对修改关闭。为此，我们将我们的类呈现为一个“黑盒”,看看我们是否可以在这种情况下改变它的行为。*

## 替代原理芭芭拉·利斯科夫(Liskov substitution)

> *程序中的对象可以被它们的继承者替换，而不改变程序的属性。为此，我们检查是否加强了前置条件，削弱了后置条件。如果出现这种情况，那么原则就没有得到尊重。*

## 界面分离

> 许多专用接口比一个通用接口要好。我们检查接口包含了多少方法，以及不同的函数是如何叠加在这些方法上的，如果有必要，我们会中断接口。

## 从属倒置原则

> *依赖应该建立在抽象上，而不是细节上。“我们检查这些类是否依赖于其他类(直接实例化其他类的对象，等等。)而如果这种依赖发生了，我们就用对抽象的依赖来代替。*

【https://it.badykov.com】原载于 2020 年 3 月 14 日[](https://it.badykov.com/blog/2020/03/14/solid-principles/)**。**