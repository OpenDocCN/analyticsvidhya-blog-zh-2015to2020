# 该算法在 P 时间内解决了 NP 完全问题

> 原文：<https://medium.com/analytics-vidhya/this-algorithm-solves-np-complete-problems-in-p-time-90f6fa938521?source=collection_archive---------18----------------------->

[布尔可满足性问题](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem)是写一个算法来决定一个给定的 N 个变量的布尔表达式是否可以为这 N 个变量中的某个选择取值“真”。

这个问题在 [NP-Complete](https://en.wikipedia.org/wiki/NP-completeness) 类。这意味着，如果有可能编写一个在 P 时间内完成的算法，那么就有一个过程来修改该算法，以便在多项式时间内解决 n P 类中的任何其他问题。

我刚刚写了一个简短的 javascript 程序，给定任何带有 *N* 个变量的布尔表达式，它在 javascript 时钟的 *N* 个“滴答”中完成并做出决定。

由于运行这个算法需要 javascript 引擎点击 N 次，所以效率是 O(N)，也就是多项式时间。

我将解释这个算法，做一些评论，并张贴代码，这样就可以清楚地知道我在做什么。

*   创建两个保存数组的对象，将一个初始化为[true]，另一个初始化为[false]
*   在每个时间步中，自我复制最近创建的对象两次，并将 true、false 附加到数组中
*   在 N 个时间步之后，布尔变量的每一组可能的值都将存在内存中
*   如果对象检测到它有正确的长度来进行检查，那么检查它并更新一个全局变量
*   因为我使用了自复制对象，所以在您检查了单个时间步骤中的每个案例*之后，终止程序*

从实用的角度来看，这个算法需要 n 次点击，这意味着它需要 n 步来检查所有的 2^N 案例，其中每一步都是 javascript 引擎的“一次点击”。

出于实际考虑，任何可以归结为所谓的布尔可满足性问题的问题现在都可以通过在多项式时间内完成的算法来解决。我不知道这是否对纯数学或 P 对 NP 问题有任何意义，但它对于编写运行速度更快的代码非常有用！

这是代码。您可以将它复制/粘贴到一个文本文件中，并将其保存为. html 文件，在浏览器中打开该文件，然后键入任何包含变量 b1、b2、…的布尔表达式。只要您将所有变量命名为 b + number，您就可以编写任何 javascript 布尔表达式，如“b1 && b2 || b3 ”,然后单击“评估”,布尔可满足性问题的答案将出现在控制台日志中

```
<!DOCTYPE html>
<html>
<head>
 <title>Boolean satisfiability problem</title>
 <style type="text/css">
  input { width: 100%; padding: 4px; font-size: 16px; }
 </style>
</head>
<body>
 <p>Type a Boolean expression in javascript using variables b1, b2, b3, ... </p>
 <input id="expression" type="text" name="expression" />
 <button id="evaluate">Evaluate</button><script type="text/javascript">
  document.getElementById('evaluate').onclick = function() {
   var isSatisfiable = false,
    numChecked = 0,
    val = document.getElementById('expression').value,
    match = val.match(/b[0-9]*/g);

   if ( !match.length ) {
    console.log('no match!');
    return;
   }var totalCases = Math.pow(2, match.length);function BioObject(value) {
    if ( value.length === match.length ) {
     var params = {};
     match.forEach(function(item, idx) {
      params[item] = value[idx];
     }); with (params) {
      if ( eval(val) ) {
       isSatisfiable = true
      } if ( ++numChecked >= totalCases ) {
        if ( isSatisfiable ) {
          console.log('is satisfiabile');
        } else {
          console.log('cannot be satisfied');
        }
      }
     }
    } else {
     setTimeout(function() {
       var t = value.slice(),
        f = value.slice();

       t.push(true)
       f.push(false) new BioObject(t)
       new BioObject(f)
     }, 1)
    }
   } new BioObject([true]);
   new BioObject([false]);
 }
 </script>
</body>
</html>
```