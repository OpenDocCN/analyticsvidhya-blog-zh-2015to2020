# Java 中的生成器设计模式

> 原文：<https://medium.com/analytics-vidhya/builder-design-pattern-in-java-f55eb6723dc4?source=collection_archive---------16----------------------->

正如其他设计模式一样，**构建器** **模式**非常有用，可以帮助程序员写出更好的代码。维基百科上说:*‘构建者模式是一种对象创建软件设计模式，目的是找到伸缩构造者反模式的解决方案。’*

好的……那么这对我改进代码有什么帮助呢？我想这是你第一个想到的东西。为了实现这个原则的力量，我将给你一些真实的例子。

想象一下，你正在吃快餐，你想吃三明治。你可以选择你想包含的内容。例如，我喜欢没有黄瓜和洋葱的三明治。

![](img/425ce38f9ba48976481a1d1a06c6e93a.png)

```
public class SandwichOrder {

    private String bread;
    private String dressing;
    private String meat;
    private String onion;
    private String cucumber;
    private String salad;
    private String tomatoes;

    public SandwichOrder(SandwichBuilder sandwichBuilder) {
        this.bread = sandwichBuilder.bread;
        this.dressing = sandwichBuilder.dressing;
        this.meat = sandwichBuilder.meat;
        this.onion = sandwichBuilder.onion;
        this.cucumber = sandwichBuilder.cucumber;
        this.salad = sandwichBuilder.salad;
        this.tomatoes = sandwichBuilder.tomatoes;
    }+ getters and setters

}
```

现在让我们用这些点菜:**面包、调料、肉、沙拉**

用编程术语来说，这意味着我们必须创建一个包含这些成分的 **SandwichOrder** 类。为了实现这一点，我们的 **SandwichOrder** 类需要一个包含以下四个字段的构造函数:

```
public SandwichOrder(String bread, String dressing, String meat, String salad) {
    this.bread = bread;
    this.dressing = dressing;
    this.meat = meat;
    this.salad = salad;
}
```

等等…这不是一个好方法。为什么！？因为，这意味着我们必须为所有的成分创建一个构造函数。也许我不喜欢洋葱，但是别人喜欢。另一个是纯素食主义者..有很多可能性。

这里是我们的地方，建筑者来帮助我们。

![](img/6aa57cbffe39d08ca08d8b93d685312e.png)

让我们创建一个 **SandwichBuilder** 类，其中我们将拥有与 **SandwichOrder** 类相同的字段。这个类也将有一个构建方法来管理三明治的准备。

```
public static class SandwichBuilder {
    private String bread;
    private String dressing;
    private String meat;
    private String onion;
    private String cucumber;
    private String salad;
    private String tomatoes;

    public SandwichOrder build() {
        return new SandwichOrder(this);
    }

}
```

现在 **SandwichOrder** 类将不再需要为每个成分组合都有构造函数，它将只有一个接受 **SandwichBuilder** 的构造函数

```
public SandwichOrder(SandwichBuilder sandwichBuilder) {
    this.bread = sandwichBuilder.bread;
    this.dressing = sandwichBuilder.dressing;
    this.meat = sandwichBuilder.meat;
    this.onion = sandwichBuilder.onion;
    this.cucumber = sandwichBuilder.cucumber;
    this.salad = sandwichBuilder.salad;
    this.tomatoes = sandwichBuilder.tomatoes;
}
```

让我们在一个 Main 方法中运行它，看看它是如何工作的。

```
public static void main(String[] args) {

    SandwichOrder.SandwichBuilder sandwichBuilder = new SandwichOrder.SandwichBuilder();

    sandwichBuilder.setBread("Bread");
    sandwichBuilder.setCucumber("Cucumber");
    sandwichBuilder.setMeat("Meat");

    SandwichOrder sandwichOrder = sandwichBuilder.build();

    System.*out*.println(sandwichOrder.toString());
}
```

以下是完整的代码:

```
package builder;

public class SandwichOrder {

    public static class SandwichBuilder {
        private String bread;
        private String dressing;
        private String meat;
        private String onion;
        private String cucumber;
        private String salad;
        private String tomatoes;

        public SandwichOrder build() {
            return new SandwichOrder(this);
        }

        public void setBread(String bread) {
            this.bread = bread;
        }

        public void setDressing(String dressing) {
            this.dressing = dressing;
        }

        public void setMeat(String meat) {
            this.meat = meat;
        }

        public void setOnion(String onion) {
            this.onion = onion;
        }

        public void setCucumber(String cucumber) {
            this.cucumber = cucumber;
        }

        public void setSalad(String salad) {
            this.salad = salad;
        }

        public void setTomatoes(String tomatoes) {
            this.tomatoes = tomatoes;
        }
    }

    private String bread;
    private String dressing;
    private String meat;
    private String onion;
    private String cucumber;
    private String salad;
    private String tomatoes;

    public SandwichOrder(SandwichBuilder sandwichBuilder) {
        this.bread = sandwichBuilder.bread;
        this.dressing = sandwichBuilder.dressing;
        this.meat = sandwichBuilder.meat;
        this.onion = sandwichBuilder.onion;
        this.cucumber = sandwichBuilder.cucumber;
        this.salad = sandwichBuilder.salad;
        this.tomatoes = sandwichBuilder.tomatoes;
    }

    public SandwichOrder(String bread, String dressing, String meat, String salad) {
        this.bread = bread;
        this.dressing = dressing;
        this.meat = meat;
        this.salad = salad;
    }

    public String getBread() {
        return bread;
    }

    public void setBread(String bread) {
        this.bread = bread;
    }

    public String getDressing() {
        return dressing;
    }

    public void setDressing(String dressing) {
        this.dressing = dressing;
    }

    public String getMeat() {
        return meat;
    }

    public void setMeat(String meat) {
        this.meat = meat;
    }

    public String getOnion() {
        return onion;
    }

    public void setOnion(String onion) {
        this.onion = onion;
    }

    public String getCucumber() {
        return cucumber;
    }

    public void setCucumber(String cucumber) {
        this.cucumber = cucumber;
    }

    public String getSalad() {
        return salad;
    }

    public void setSalad(String salad) {
        this.salad = salad;
    }

    public String getTomatoes() {
        return tomatoes;
    }

    public void setTomatoes(String tomatoes) {
        this.tomatoes = tomatoes;
    }

    @Override
    public String toString() {
        return "SandwichOrder{" +
                "bread='" + bread + '\'' +
                ", dressing='" + dressing + '\'' +
                ", meat='" + meat + '\'' +
                ", onion='" + onion + '\'' +
                ", cucumber='" + cucumber + '\'' +
                ", salad='" + salad + '\'' +
                ", tomatoes='" + tomatoes + '\'' +
                '}';
    }
}
```

希望这一行能帮助你，并在你未来的应用中找到用途。

干杯，

席尔武·拉达。