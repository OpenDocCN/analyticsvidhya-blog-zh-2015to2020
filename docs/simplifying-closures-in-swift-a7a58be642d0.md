# 简化 Swift 中的闭包

> 原文：<https://medium.com/analytics-vidhya/simplifying-closures-in-swift-a7a58be642d0?source=collection_archive---------18----------------------->

别担心，你并不孤单。闭包可能会令人困惑，如果您和我一样，您可能已经阅读了各种定义和帖子，试图找出它们。但是读完之后，你会更加困惑。希望这能帮助你筛选技术术语，让你有更好的理解。

# 什么是终结？

只是一个有一些特殊规则的函数，写的有点不一样。

## *语法和规则*

```
{(parameters) -> return type in
    code
}
```

**规则 1:** 如果没有参数，可以省略箭头操作符和返回类型。也可以用下划线代替参数名。

```
let closure = { () in
    return "This is still a valid closure"
}()
```

**规则二:**如果参数类型是编译器已知的，可以省略类型。

```
let closure = {(str) in
    return "This is a \(str) in a closure"
}closure("string")
```

**规则 3:** 如果匿名函数不带参数，并且返回类型可以省略，就不必使用‘in’表达式。但是，如果有参数，您仍然可以通过使用$0 语法省略名称。

**规则四:**如果闭包是一行，可以省略关键字 return。

```
let closure = {
    "No return keyword necessary"
}
```

**规则 5:** 如果省略了参数类型，则可以省略参数列表两边的括号。

```
let closure = { str in
    return "This is how you omit the parentheses around \(str)"
}
```

**规则 6:** 如果函数的最后一个参数是闭包，可以使用尾随闭包语法，这样可以省略函数标签和/或函数括号。

```
func doSomething(_ closure: () -> ()) {
    closure()
}doSomething { //Omitting ()
    "This is a trailing closure, notice there is no function call"
}
```

> 专业提示:如果一个匿名函数接受参数，你必须确认所有的参数。要么使用它们，要么用下划线忽略它们。

# **使用闭包**

**规则 1** :就像函数一样，你可以给变量分配闭包，稍后或者立即调用它们。

```
let closure = {
    return "This is a closure"
}print(closure()) //prints "This is a closure"func f() -> String {
    return "Functions look pretty similar to closures, right?"
}let varFunc = f
print(varFunc())
```

*注意:你也可以定义然后调用闭包。请习惯这一点，因为这是初始化实例属性时非常常见的模式。*

```
let closure = {
    return "This is a closure"
}()
```

规则 2 :闭包不使用参数标签。

```
let printThis = { (str: String) in
    print("This is a \(str)")
}
printThis("string") //vs func equivalent printThis(str: "string")
```

规则 3:在任何使用变量的地方，你都可以用闭包来代替。通常，函数有参数，比如整型、字符串或数组。在下面的例子中，我们使用闭包 printThis 作为新函数 closureAsParam 中的变量。有几件事情正在发生，所以让我们把它分解成它的组件。我们有一个新的函数 closureAsParam，它接受一个闭包。这个闭包参数也接受一个字符串并返回 void，表示为(String) - > Void。我们从函数中调用闭包，并向它提供字符串“插入值”。

所以当我们调用这个函数 closureAsParam 时，我们可以提供任何接受一个字符串并返回 void 的闭包。在我们的例子中，我们使用闭包 printThis，有效地调用 printThis(“插入的值”)。

```
func closureAsParam(closureParam: (String) -> Void){
    closureParam("inserted value")
}closureAsParam(closureParam: printThis)
```

*注意:你可以用两个括号()代替 Void*

**规则 4** :如果一个函数的最后一个参数是闭包，你可以在花括号中的函数调用之后传递你的闭包。这就是所谓的尾随闭包。假设您想要更改 closureAsParam 函数，但不想直接修改该函数。

```
closureAsParam() { _ in
    printThis("different value") //'This is a different value'
}
```

**规则 5** :闭包捕获对外部变量的引用，并可以在其中存储值。当你把一个函数/闭包赋给一个变量时，你设置了这个变量被这个函数/闭包引用。

```
class someClass {
    var someVar = "This is the original value"
    func printVar() {
        print(self.someVar)
    }
}func runFunc(_ f:() -> ()) {
    f()
}let ex = someClass()
ex.printVar() //"This is the original value"
ex.someVar = "This is a new value"let ex2 = ex.printVar
runFunc(ex2) //"This is a new value"
```

在这个例子中，我们创建了一个 someClass 的实例，并将 someVar 设置为一个新值。然后我们将 ex2 设置为函数 printVar。我们不显式调用 ex.printVar()，而是将函数作为值传递。所以当我们运行函数 runFunc 时，我们可以访问一个 someVar 的引用。因此，如果我们更改 ex.someVar，当我们运行 runFunc 时，它也会反映所有新的更改！！

```
let ex = someClass()
let ex2 = ex.printVarrunFunc(ex2) //"This is the original value"
ex.someVar = "New Value"runFunc(ex2) //"New Value"
```

我们没有改变 ex2，刚刚发生了什么？ex1 和 ex2 都引用了 someClass 的同一个实例！

让我们看看另一个捕获闭包内的值的例子:今天你和同事打了几次招呼？

```
func greeting () -> (String) -> (){
    var greetings = 0 return { name in
        print("Hey \(name), how about that weather though?")
        greetings +=1
    }}let hi = greeting()
hi("Vincent") //Greetings = 1
hi("Tiffany") //Greetings = 2
hi("Brian") //Greetings = 3
```

即使 greetings 在该函数的范围内，我们的闭包仍然有对它的引用。

# 闭包的用例

1.  **代码更短** 如果一个函数接受另一个函数，尽量用闭包代替。

```
var nums = [1,2,3,4,5]
evenNumsFirst = nums.sorted(by: {num1 % 2 < num2 % 2 })evenNumsFirst //[2, 4, 1, 3, 5]
```

2.**你有一个返回函数的函数(高阶函数)** 在考虑什么时候使用闭包的时候，永远记住 DRY 原则。任何时候你有硬编码的值，或者重复的代码，这可能是一个重构代码的好机会！

# **接下来是什么？**

如果您刚刚开始学习 swift，那么使用闭包还有很多内容！首先，掌握使用闭包的基本知识。然后，在继续学习转义和自动关闭之前，尝试在代码中使用它们。

和往常一样，如果你想深入了解，我建议你通读文档:[https://docs . swift . org/swift-book/language guide/closures . html](https://docs.swift.org/swift-book/LanguageGuide/Closures.html)