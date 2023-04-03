# 简单英语的反应—状态

> 原文：<https://medium.com/analytics-vidhya/reactjs-in-simple-english-state-7aecb126a29b?source=collection_archive---------16----------------------->

# 什么是状态？

***状态*** 是你 app 中的属性。应用程序运行时，根据业务逻辑改变 ***状态*** 以满足用户需求。

如果你熟悉 Java， ***state*** 有点像私有变量，我们不能在创建它的类之外编辑它。它们对其他类是不可见的，除非我们显式地传递给另一个类(或函数组件),在那里它们被称为“道具”。只有显式地传递函数，其他类才能编辑 ***状态***

在 React 16.8 之前， ***状态*** 只在类组件中可用。从此，函数组件有了自己的 ***状态*** 通过 React 钩子使用它。

在本文中，我们将重点关注类组件中的**状态**。请随意尝试页面底部的工作示例

1.  初始化状态

```
class App extends Component {
  constructor (props) {
    super(props);
    this.state = {
    car: 
      { model: 'tesla modal-X', price: 50000 },

      otherState: 'some other value'
    }; }
}export default App;
```

你可以在构造函数内部或外部初始化状态，它们是一样的，只是编码方式不同。

```
class App extends Component {
  state = {
    car: 
      { model: 'tesla modal-X', price: 50000 },

    otherState: 'some other value'
  };
}export default App;
```

2.将状态传递给其他类/函数

假设我们有一个名为" Car "的类，我们想在类" App "的 render()函数中将状态从类" App "传递给类" Car"
，我们通过*price*
*this . state . Car . model*通过*model**this . switch Car handler*通过*单击*
* bec

```
render() {
  return (
    <div className="App">
      <div>
        <p>
          <Car 
            price={this.state.car.price}
            model={this.state.car.model}
            clicked={this.switchCarHandler}
          ></Car>
        </p>
      </div>
    </div>
  );
}
```

在类" Car"
中，我们通过*this . props . price*
*this . state . Car . model*通过*this . props . model*
*this . switch Car handler*通过 *this.props.clicked*

```
import React, { Component } from "react";class Car extends Component {
  render() {
    console.log("[Persons.js] rendering...");return (
      <div>
        <button onClick={this.props.clicked}>Switch Car</button>
        <p>
          {this.props.model} - ${this.props.price}
        </p>
      </div>
    );
  }
}
export default Car;
```

3. ***状态*** 是不可变的

我们不改变 ***状态*** ，而是替换它。

考虑下面的例子
我们试图通过一个按钮触发“switchCarHandler”来改变处于 ***状态*** 的汽车对象。在“switchCarHandler”中，我们直接更新对象内部的值。

```
//Not working exampleimport React, { Component } from 'react';
import './styles.css';class App extends Component {
  state = {
    car: { model: "tesla modal-X", price: 50000 },
    insuranceFee: 1000
};switchCarHandler = () => {
    this.state.car.modal = 'toyota';
  };render() {
    return (
      <div className="App">
        <button onClick={this.switchCarHandler}>Switch Car</button>
        <div>
            <p>Spent {this.state.car.price} for my new car - {this.state.car.model}</p>           
        </div>
      </div>
    );
  }
}export default App;
```

这不起作用，因为只有 **setState()** 可以触发 render 方法。更新状态的正确方法是创建并用另一个对象替换

```
switchCarHandler = () => {
   this.setState({
      car: { model: "tesla modal-S", price: 40000 }
   });
};
```

***注意:this.setState 只替换" car "对象，它仍然保留 insuranceFee***

4. *setState()* 是异步

这意味着状态的改变可能不会立即生效，因此状态值可能不是最新的
假设我们要将保险费的价格提高 10%,但这是我们的特殊客户，所以我们提供 10%的折扣

考虑这个例子，首先我们通过 calculateInsuranceFee 函数增加 10%,然后通过 calculateDiscount 函数提供 10%的折扣。如果原价是 1000，那么

1000* 1.1 * 0.9 = 990

然而，我们只得到 900，因为我们在两个函数中都引用了 *this.state.insuranceFee* 。如上所述，React 不保证是最新的，因此在这种情况下 React 可能忽略了 calculateInsuranceFee 的结果，并使用原始值-1000，1000*0.9 计算折扣，因此结果是 900

```
//Not working examplestate = {
  car: { model: "tesla modal-X", price: 50000 },
  insuranceFee: 1000
};calculateInsuranceFee = () => {
  this.setState({ insuranceFee: this.state.insuranceFee * 1.1 });
};calculateDiscount = () => {
  this.setState({ insuranceFee: this.state.insuranceFee * 0.9 });
};increasePriceHandler = () => {
  this.calculateInsuranceFee();
  this.calculateDiscount();
};
```

为了解决这个问题，React 提供了另一种形式的 setState()

```
calcInsuranceFee() {
  this.setState((state) => {
    return { insuranceFee: state.insuranceFee * 1.1 };
  });
}calcDiscount() {
  this.setState((state) => {
  return { insuranceFee: state.insuranceFee * 0.9 };
  });
}increasePriceHandler = () => {
  this.calculateInsuranceFee();
  this.calculateDiscount();
};
```

问题已修复，它将输出 990 而不是 900

对于上面的例子，尝试工作代码