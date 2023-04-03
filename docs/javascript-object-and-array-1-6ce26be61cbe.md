# ES6 Javascript 对象和数组(1)

> 原文：<https://medium.com/analytics-vidhya/javascript-object-and-array-1-6ce26be61cbe?source=collection_archive---------15----------------------->

在 Javascript 中，有很多与对象和数组相关的方法，初学者要记住它们并不容易。

在本节中，我将向您介绍 object . keys()/Array map()/[…](spread operator)以及与它们相关的一个示例。一旦你理解了这些例子，我相信你会理解并记住它们。

Javascript 中的对象
用花括号开始和结束

* * *不要在 Javascript 对象中使用箭头函数，详细内容请阅读[ES6 Javascript—“this”关键字](/@fifithecat/es6-javascript-this-keyword-4377c61f6f2d)

```
let car = {type:"Tesla", model:"Modal X", color:"white"};//In Javascript, objects can contains function
let car = {
    type:"Tesla", 
    model:"Modal X", 
    color:"white",
    summary: function() {
       console.log(this.type, this.model, this.color);
    }
};
```

Javascript 中的数组
以括号开始和结束

```
//Various ways to create new arraylet cars = ["Tesla", "Toyota", "BMW"];let cars = new Array("Tesla", "Toyota", "BMW");let cars = new Array(3); //empty array with fixed length of 3let cars = Array(3); //empty array with fixed length of 3let cars = Array.of("Tesla", "Toyota", "BMW");let cars = Array.from("Tesla");//convert iterable ro array-like objects to array
//["T","e","s","l","a"]
```

对象.键

```
Retrieve keys from object and place them in arraylet car = {type:"Tesla", model:"Modal X", color:"white"};
const carData = Object.keys(car)
console.log(carData)//it prints 
//["type", "model", "color"]What if we put array inside Object.keys?let cars = ["Tesla", "Toyota", "BMW"];
const carData = Object.keys(cars)
console.log(carData)//It would return the array index
//["0", "1", "2"]
```

数组映射()

这是一个基于 map()中现有数组
中的每个元素返回一个新数组的函数，你必须传递一个回调函数——这个函数将为每个数组元素执行。
第一个参数——当前元素，由您指定的变量名。它表示 prices 数组 10.99，5.99，3.99，6.59 中的元素
第二个参数—索引，当前数组的索引，由您分配的变量名
第三个参数—我们要处理的数组

在我们的例子中，我们将计算 15%商品及服务税的价格，新价格将是当前价格* 1.15，因此价格*(1+商品及服务税)在返回声明中

```
const prices = [10.99, 5.99, 3.99, 6.59];
const gst = 0.15;const pricesWithGST = prices.map((price, idx, prices) => {
      return price * (1+gst);
});
console.log(pricesWithGST);//it prints
//[12.638499999999999, 6.8885, 4.5885, 7.578499999999999]
//the prices now include 15% of GST
```

[…]扩展运算符

这是一个有助于从数组中提取数据的语法，就像从数组
***【10.99，5.99，*** 10.99，5.99，3.99，6.59***】***10.99，5.99，3.99，6.59
*【约翰】，【菲菲】*

```
**//Copy array data to new array
const names = ['John', 'Fifi'];
const copiedNames = [...names];
console.log(copiedNames);//["John", "Fifi"]//Append data from other array to existing array
const moreNames = ['Eunice', ...names];
console.log(moreNames);//["Eunice", "John", "Fifi"]//extract data from array and place in function accept parameters with comma separator
const prices = [14.99, 7.99, 3.99, 2.59];
console.log(Math.min(...prices))//copy array contains object
const cars = [{ brand: 'Tesla', color: 'white' }, { brand: 'BMW', color: 'black' }];
const copiedCars = [ ...cars];console.log(copiedCars);//new array but same objects inside
//[{brand: "Tesla", color: "white"},{brand: "BMW", color: "black"}]It copies from cars array and save in new array copiedCars, but since the elements inside cars are object, so that means the elements inside copiedCars are the same objects. That means if you edit the elements inside the copiedCars, the elments inside cars will be affected too, unless you handled manually**
```

**对于新的学习者，很难记住所有的，让我们做一些练习，试着理解下面的代码片段**

```
**const fruits   = {
        apple:1,
        orange:1,
        peach:2,
        pear:2

    }const FruitsTags = Object.keys(fruits)
    .map(fruitKey => {
        return [...Array(fruits[fruitKey])].map(
            (_, i)=> {return `<Fruit key=${fruitKey + i} type=${fruitKey}/>`}

        );   
    });

console.log(FruitsTags);**
```

**解释—循序渐进**

**第一步，很简单，只需从对象中提取那些键**

```
 **const FruitsTags = Object.keys(fruits )console.log(FruitsTags );
["apple", "orange", "peach", "pear"]**
```

**在 map 中，它将创建一个空数组，其长度基于基于水果数组的每个值(1，1，2，2)。开始时使用 spread 操作符，它将从每个数组中提取那些空值，并将其打包回数组。这样做的原因是因为空数组没有我们下一步需要的任何索引。通过使用 spread 运算符对空数组进行重新打包，它将替换为一个未定义的值，从而得到索引**

```
**const FruitsTags = Object.keys(fruits)
.map(fruitKey => {
        return [...Array(fruits[fruitKey])]   
});console.log(FruitsTags );
[[undefined]
[undefined]
[undefined, undefined]
[undefined, undefined]]**
```

**在嵌套映射中，我们只需在每个数组中添加键和索引就可以返回一个字符串**

```
**const FruitsTags = Object.keys(fruits)
.map(fruitKey => {
    return [...Array(fruits[fruitKey])].map(

                (_, i)=> { return (fruitKey + i)}

    );   
});console.log(FruitsTags);["apple0"]
["orange0"]
["peach0", "peach1"]
["pear0", "pear1"]**
```

**使用反勾号(模板文字)，我们可以将表达式括在里面并打印标签。实际上，这是来自一个关于如何根据 object 中的值动态生成组件的 React 示例。**

```
**const FruitsTags = Object.keys(fruits)
    .map(fruitKey => {
        return [...Array(fruits[fruitKey])].map(

            (_, i)=> {return `<Fruit key=${fruitKey + i} type=${fruitKey}/>`}

        );   
    });

console.log(FruitsTags);
[["<Fruit key=apple0 type=apple/>"]
["<Fruit key=orange0 type=orange/>"]
["<Fruit key=peach0 type=peach/>", "<Fruit key=peach1 type=peach/>"]
["<Fruit key=pear0 type=pear/>", "<Fruit key=pear1 type=pear/>"]]**
```

**如果你花时间在你的控制台上尝试过，我想你现在应该很理解 object . keys()/Array map()/[…](spread operator)了。**