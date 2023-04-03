# ES6 JavaScript—“this”关键字

> 原文：<https://medium.com/analytics-vidhya/es6-javascript-this-keyword-4377c61f6f2d?source=collection_archive---------19----------------------->

试举例总结“this”的用法

*   每一个用 function 关键字或速记方法创建的函数都有自己的“this ”,它通常绑定到调用它的“who”。
*   箭头函数不会将“这个”绑定到任何东西，里面的“这个”依赖于周围的上下文

示例 1

错误函数在这里声明了“handler ”,这意味着“this”什么都不是，就像在处理程序之外一样，因此输出是窗口对象

```
const handler = () => {
 console.log(this);
};
btn1.addEventListener('click', handler);
```

输出

```
Window {parent: Window, opener: null, top: Window, length: 0, frames: Window, …}
```

示例 2

这里用“function”关键字，“this”绑定到某个东西。在这种情况下，浏览器自动将" this "绑定到按钮，因为按钮触发了调用

```
const handler = function() {
 console.log(this);
};
btn.addEventListener('click', handler);
```

输出

```
<button id="btn1"></button>
```

示例 3

为什么我们需要“这个”

```
const name = 'Johnny';

const person = {
   name: 'Fifi',
   greet() {
      console.log(name);        
      console.log(this.name);   
   }
};

person.greet();
```

输出

```
Johnny
Fifi
```

实例 4

正如我们所称的“newMovie.getFormattedTitle()”，getFormattedTitle()绑定到 newMovie 上下文。然而，在 oldMovie 中，getFormattedTitle()是用 arrow 函数声明的，这意味着它没有绑定到任何东西，而是绑定到周围的上下文。在这种情况下，周围的上下文不在 oldMovie 内部，因为我们不能在那里放置任何“this ”,相反，周围的上下文应该在 oldMovie 外部。于是我们打印了 Window 对象，无法调用 old movie . getformattedditle()；这就是为什么我们通常不在对象内部使用箭头函数(在内部函数中，情况不同)。

```
const newMovie = {
    info: {
      title:'007 No time to die'
    },
    getFormattedTitle() {
      console.log (this.info.title.toUpperCase());
    }
  };
const oldMovie = {
    info: {
      title:'Saving private Ryan'
    },
    getFormattedTitle: ()=> {
      console.log(this);
      console.log(this.info.title.toUpperCase());
    }
  };newMovie.getFormattedTitle();
oldMovie.getFormattedTitle();
```

输出

```
007 NO TIME TO DIE
Window {parent: Window, opener: null, top: Window, length: 0, frames: Window, …}
Uncaught TypeError: Cannot read property 'title' of undefined
```

实例 5

对象的内部函数不绑定到对象本身，而是绑定到窗口

```
const newTeam = {teamName: 'ES6', 
 people: ['Johnny', 'Fifi'], getTeamMembers() {
  this.people.forEach(p => {
   console.log(p + ' - ' +this.teamName); 
 });
}};const oldTeam = {teamName: 'Java', 
 people: ['Johnny', 'Fifi'], getTeamMembers() {
  this.people.forEach(function(p) {
  console.log(this);
  console.log(p + ' - ' + this.teamName); 
 });
}};newTeam.getTeamMembers();
oldTeam.getTeamMembers();
```

输出

```
Johnny - ES6
Fifi - ES6
Window {parent: Window, opener: null, top: Window, length: 4, frames: Window, …}
Johnny - undefined
Window {parent: Window, opener: null, top: Window, length: 4, frames: Window, …}
Fifi - undefined
```

实例 6

在类中声明函数或在回调中调用函数时，使用箭头函数总是安全的。

```
class Greeting {constructor() {
    this.name = 'John';
  }greetMorning() {console.log('Good Morning ' + this.name);}greetAfternoon = () => {console.log('Good Afternoon ' + this.name);};

greetEvening = function() {console.log('Good Evening ' + this.name);}  
}const g = new Greeting();g.name = "fifi"g.greetMorning();
g.greetAfternoon();
g.greetEvening();const button1 = document.getElementById('btn1');
const button2 = document.getElementById('btn2');
const button3 = document.getElementById('btn3');
const button4 = document.getElementById('btn4');
const button5 = document.getElementById('btn5');
const button6 = document.getElementById('btn6');button1.addEventListener('click', () => (g.greetMorning()));
button2.addEventListener('click', () => (g.greetAfternoon()));
button3.addEventListener('click', () => (g.greetEvening()));
button4.addEventListener('click', g.greetMorning);
button5.addEventListener('click', g.greetAfternoon);
button6.addEventListener('click', g.greetEvening);
```

输出

```
Good Morning fifi
Good Afternoon fifi
Good Evening fifi
Good Morning fifi
Good Afternoon fifi
Good Evening fifi
Good Morning 
Good Afternoon fifi
Good Evening
```