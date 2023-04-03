# 有用的 ES6 JavaScript 函数——map()、filter()、reduce()、…

> 原文：<https://medium.com/analytics-vidhya/useful-javascript-functions-map-filter-reduce-a06e8b446990?source=collection_archive---------24----------------------->

ES6 版本的 JavaScript 附带了各种方便的方法。我在下面列出了其中的一些细节。

# *箭头功能*

箭头函数是在 JavaScript 中编写函数的一种更短更快的方式，不需要使用*函数关键字。*

```
const a = (x, y) => {
 let z = x + y
 return z
}let b = a(5, 7) // Returns 12
```

这里， *x* 和 *y* 是传递给函数的参数，函数处理这些参数并返回总和。也可以用更短的方式写。

```
(x, y) => x + y
```

这是一个单行函数。不需要大括号和*返回*关键字。又短又简单对吧！！！

# 地图()

map()函数遍历一个 iterable 对象并返回一个列表。例如，有一个包含姓名和电话号码的联系人列表，您只需要获取列表中的姓名。

```
let addressBook = [{
 name: "Ben",
 phoneNumber: "1234567890"
},{
 name: "Robin",
 phoneNumber: "9876543210"
}]let names = addressBook.map(contact => contact.name)
console.log(names)  // ['Ben',  'Robin']
```

# 过滤器()

比方说，你想从列表中过滤一些元素。代替传统的使用*进行*循环的方式，可以使用 *filter()* 函数。

例如，您有一个雇员及其工资的列表。您只想检索那些收入超过 10，000 美元的雇员。

```
let employees = [{
 id: 1,
 name: 'Ben',
 salary: 12000
},{
 id: 2,
 name: 'Jack',
 salary: 5000
}, {
 id: 3,
 name: 'James',
 salary: 14000
}]let highlyPaidEmployees = employees.filter(emp => emp.salary > 10000)
console.log(highlyPaidEmployees)// Returns the following list
[{ id: 1, name: 'Ben', salary: 12000 },{ id: 3, name: 'James', salary: 14000 }]
```

> map()和 filter()函数不会在同一个列表中发生变化。他们处理列表并返回另一个列表。

# 减少()

*reduce()* 函数采用带有两个参数累加器和列表项的函数。与 *map()* 和 *filter()* 不同，它返回单个值。

让我们从上面的例子中取出雇员列表。要获得所有员工的工资总额:

```
let totalSalary = employees.reduce((accumulator, emp) => accumulator + emp.salary, 0)console.log(totalSalary) // Returns 31000
```

注意，我传递给 reduce()函数的第二个参数是 0。这是累加器的起始值。

# 解构对象

通过使用分解结构，我们可以从数组中解包值，或者从对象中解包属性。

```
let employee = {id: 1, name: 'Ben'}
let {id, name} = employee
console.log(name) // Ben
console.log(id) // 1let [a, b] = [1, 2]
console.log(a)  // 1
console.log(b)  // 2
```

# 传播和休息运算符

*spread* 操作符实际上是“传播”数组中的值。下面来看看。

```
let a = [1, 2, 3]
let b = [4, 5]
let mergedArray = [...a, ...b]  // [1, 2, 3, 4, 5]
```

*rest* 参数将所有元素收集到一个数组中。

```
function concatenate(...args){
 let name = ''
 for (let arg of args) name += ' ' + arg;
 return name
}concatenate('Raymond', 'Reddington')  // Prints 'Raymond Reddington'concatenate('Raymond', 'Red', 'Reddington')
// Prints 'Raymond Red Reddington'
```

在上面的例子中， *concatenate()* 函数接受任意数量的参数。rest 运算符将所有值收集到一个数组中。

试着在你的项目中使用这些函数。会很有趣的。

继续编码！