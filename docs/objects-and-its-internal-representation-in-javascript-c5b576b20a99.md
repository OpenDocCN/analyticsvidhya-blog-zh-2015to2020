# 对象及其在 JavaScript 中的内部表示

> 原文：<https://medium.com/analytics-vidhya/objects-and-its-internal-representation-in-javascript-c5b576b20a99?source=collection_archive---------3----------------------->

在 JavaScript 中，对象是最重要的数据类型，是现代 JavaScript 的组成部分。这些对象与 JavaScript 的原始数据类型(数字、字符串、布尔、空、未定义和符号)有很大不同，因为这些原始数据类型都存储单个值(取决于它们的类型)。

对象更复杂，每个对象可能包含这些原始数据类型以及引用数据类型的任意组合。
一个对象，是一个引用数据类型。被赋予引用值的变量被赋予一个引用或指向该值的指针。该引用或指针指向内存中存储该对象的位置。变量实际上并不存储值。

不严格地说，JavaScript 中的对象可以被定义为相关数据、原语或引用类型的无序集合，采用“键:值”对的形式。这些键可以是变量或函数，在对象的上下文中分别称为属性和方法。

例如，如果你的对象是一个学生，那么它将拥有姓名、年龄、地址、id 等属性，以及`updateAddress`、`updateNam`等方法。

# **对象和属性**

JavaScript 对象具有与之相关联的属性。对象的属性可以解释为附加到对象上的变量。对象属性基本上和普通的 JavaScript 变量一样，除了对象的附件。对象的属性定义了对象的特征。您可以使用简单的点符号来访问对象的属性:

```
objectName.propertyName
```

像所有 JavaScript 变量一样，对象名(可以是普通变量)和属性名都是区分大小写的。您可以通过为属性赋值来定义属性。例如，让我们创建一个名为`myCar`的对象，并赋予它属性`make`、`model`和`year`，如下所示:

```
var myCar = new Object();
myCar.make = 'Ford';
myCar.model = 'Mustang';
myCar.year = 1969;
```

对象的未赋值属性是`[undefined](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/undefined)`(而不是`[null](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/null)`)。

```
myCar.color; // undefined
```

JavaScript 对象的属性也可以使用方括号符号来访问或设置(更多详细信息参见[属性访问器](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Property_Accessors))。对象有时被称为*关联数组*，因为每个属性都与一个可以用来访问它的字符串值相关联。因此，例如，您可以按如下方式访问`myCar`对象的属性:

```
myCar['make'] = 'Ford';
myCar['model'] = 'Mustang';
myCar['year'] = 1969;
```

对象属性名称可以是任何有效的 JavaScript 字符串，也可以是任何可以转换为字符串的名称，包括空字符串。但是，任何不是有效 JavaScript 标识符的属性名(例如，包含空格或连字符或以数字开头的属性名)都只能使用方括号表示法来访问。当要动态确定属性名时(直到运行时才确定属性名)，这种表示法也非常有用。示例如下:

```
// four variables are created and assigned in a single go, 
// separated by commas
var myObj = new Object(),
    str = 'myString',
    rand = Math.random(),
    obj = new Object();
myObj.type              = 'Dot syntax';
myObj['date created']   = 'String with space';
myObj[str]              = 'String value';
myObj[rand]             = 'Random Number';
myObj[obj]              = 'Object';
myObj['']               = 'Even an empty string';console.log(myObj);
```

您也可以使用存储在变量中的字符串值来访问属性:

```
var propertyName = 'make';
myCar[propertyName] = 'Ford';propertyName = 'model';
myCar[propertyName] = 'Mustang';
```

您可以使用带有`[for...in](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/for...in)`的括号符号来迭代一个对象的所有可枚举属性。为了说明这是如何工作的，当您将对象和对象的名称作为参数传递给函数时，下面的函数将显示对象的属性:

```
function showProps(obj, objName) {
  var result = ``;
  for (var i in obj) {
    // obj.hasOwnProperty() is used to filter out properties from the object's prototype chain
    if (obj.hasOwnProperty(i)) {
      result += `${objName}.${i} = ${obj[i]}\n`;
    }
  }
  return result;
}
```

因此，函数调用`showProps(myCar, "myCar")`将返回以下内容:

```
myCar.make = Ford
myCar.model = Mustang
myCar.year = 1969
```

# **用 JavaScript 创建对象:**

# 使用对象文字创建 JavaScript 对象

创建 javascript 对象最简单的方法之一是对象文字，只需在大括号中定义属性和值，如下所示

```
let bike = {name: 'SuperSport', maker:'Ducati', engine:'937cc'};
```

# 使用构造函数创建 JavaScript 对象

构造函数只不过是一个函数，在 new 关键字的帮助下，构造函数允许创建多个相同风格的对象，如下所示

```
function Vehicle(name, maker) {
   this.name = name;
   this.maker = maker;
}
let car1 = new Vehicle(’Fiesta’, 'Ford’);
let car2 = new Vehicle(’Santa Fe’, 'Hyundai’)
console.log(car1.name);    //Output: Fiesta
console.log(car2.name);    //Output: Santa Fe
```

# 使用新的 JavaScript 关键字

以下示例还创建了一个具有四个属性的新 JavaScript 对象:

例子

var person =新对象()；
person . first name = " John "；
person . last name = " Doe "；
人，年龄= 50；
person . eye color = " blue "；

# 使用`Object.create`法

也可以使用`[Object.create()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/create)`方法创建对象。这种方法非常有用，因为它允许您为要创建的对象选择原型对象，而不必定义构造函数。

```
// Animal properties and method encapsulation
var Animal = {
  type: 'Invertebrates', // Default value of properties
  displayType: function() {  // Method which will display type of Animal
    console.log(this.type);
  }
};
// Create new animal type called animal1 
var animal1 = Object.create(Animal);
animal1.displayType(); // Output:Invertebrates
// Create new animal type called Fishes
var fish = Object.create(Animal);
fish.type = 'Fishes';
fish.displayType(); // Output:Fishes
```