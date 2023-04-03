# 锈中数组的前几行

> 原文：<https://medium.com/analytics-vidhya/the-first-few-lines-of-arrays-in-rust-b5d181b357a4?source=collection_archive---------13----------------------->

![](img/295d4c2a2952415f0f9864c98e62df10.png)

paweczerwi ski 在 [Unsplash](https://unsplash.com/s/photos/array?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

> *一个固定大小的数组，表示为[T；N]，用于元素类型 T，以及非负的编译时常量大小 n*

这是你在 Rust 中阅读关于 [*数组*](https://doc.rust-lang.org/std/primitive.array.html) 类型文档的前几行时所发现的。

让我们一点一点地检查这个。
首先，我们有以下内容

> *固定大小的数组…*

好了，现在我们知道数组在内存中永远不会增长或收缩。如果你创建了一个能够容纳 5 个类型为 *bool* 的元素的数组，它将永远是一个可以容纳 5 个*bool*的数组。

我们继续寻找描述

> …表示为[T；N]

*【T；告诉我们声明数组时使用的格式。
不过是 *T* 和 *N* ？*

> …对于元素类型，T

*T* 表示数组元素的类型。如果我们想在数组中存储数字，那么 *T* 可以引用 *i32* 中的 Rust、*【i32；n】。为了更好地理解，我们需要阅读下一部分*

> …以及非负的编译时常量大小 n。

*N* 代表我们数组的大小，文档中提到了 *N* 必须遵循的两条规则。

**第一条规则** *N* 需要是非负数。让我们违反规则，看看会发生什么:-)

```
let array_with_negative_size: [i32; -3];
```

这会产生编译器错误

```
error[E0600]: cannot apply unary operator `-` to type `usize`
```

有了这个错误，我们可以看到编译器期望 *usize* 作为 *N 的类型。*类型 *usize* 代表一个无符号整数，所以它永远不能小于 *0* 。

**第二个规则** *N* 需要是一个编译时常数。如果用于 *N* 的 what ever 表达式不可替换为常数值，则规则被破坏。

这里我们尝试使用传递给函数的参数 *n* 作为 *N.* 的值

```
fn compile_time_array_size(n: usize) {
  let array_with_positive_size: [i32; n];
}
```

这会产生编译器错误

```
error[E0435]: attempt to use a non-constant value in a constant
```

现在我们有了。

# 更进一步…

> *创建数组有两种语法形式:*
> 
> *每个元素的列表，即【x，y，z】。*
> 
> *一个重复的表达式[x；N]，它产生一个包含 x 的 N 个副本的数组，x 的类型必须是 Copy。*

如果你继续阅读文档，这就是你的发现。
它强调了实际创建数组的两种方法。

## 方法**一**

> 包含每个元素的列表，即[x，y，z]。

```
let array_of_numbers = [0, 1, 1, 2, 3, 5, 8];
```

在这种情况下我们声明了什么？

首先，编译器可以计算出我们的数组中有 7 个元素，并使用它来设置数组的大小( *N)* 。

但是 *T* 呢？
Rust 对于数字默认为 *i32* ，这意味着我们刚刚声明了一个包含 7 个元素的数组 *i32s* ，*【i32；7]* 。

如果你有一个文本，其中任何元素值超过了一个 *i32* 的能力，会发生什么？

```
let array_of_numbers = [0, 1, 10_000_000_000];
```

编译器会抱怨

```
error: literal out of range for `i32`
```

由此我们可以看出，Rust 不会帮助我们自动将类型更改为更合适的类型，比如一个 *i64* 。为此，我们需要做以下工作

```
let array_of_numbers: [i64; 3] = [0, 1, 10_000_000_000];
```

我们现在有一个包含 3 个 i64 个 T21 元素的数组。

数组文本中的特定值可以用常量以外的值来描述。例如，您可以用表达式设置这些值。

```
fn array_littera(s: String) {
  let array = [s.len(), s.len() + 1, s.len() + 2];
}
```

## 方法**两个**

> 重复表达式[x；N]，它产生一个包含 x 的 N 个副本的数组，x 的类型必须是 Copy。

将我们的注意力放在数组上，我们只需浏览一下“*必须复制*”部分，并理解到——*复制*特征只是允许我们复制值的东西。更重要的部分是理解复制功能的用途。

*N* 仍然表示数组的大小，但是我们现在有了 *x，*这是一个“重复表达式”。换句话说，这将采用 *x* 并将其用作所有元素的初始值。现在我们看到了为什么它需要被复制，我们给 *x* 的值需要被重复复制到数组中。

文档中使用了“表达式”这个词，在 Rust 中我们可以用一个表达式做很多事情。这意味着我们可以更加动态地创造我们的 *x* 价值。

让我们玩一会儿这个:-)

```
fn create_array_with_fn_as_repeatable_expression(string: String) {
  let array = [string.len(); 2];
}
```

或者…你可以这样做(你可能不应该这样做)。

```
fn create_array_with_repeatable_expression(input: u128) {
  let array_of_number = [{
    let modulus = 10 * input;
    std::time::SystemTime::now()
      .duration_since(std::time::UNIX_EPOCH)
      .map(|d| d.as_millis())
      .map(|n| n % modulus)
      .unwrap_or(10)
   }; 3];
}
```

**但是为什么？** 为什么这两种方法很好了解？
一个原因是如果一个数组没有初始化，你就不能使用它。

例如，这将会失败

```
let array: [i32; 10];
let length = array.len();
```

并产生编译器错误

```
error[E0381]: borrow of possibly-uninitialized variable: `array`
```

所以这两个方法允许你初始化一个数组

*   显式设置每个值
*   或者通过提供要复制到数组的每个元素中的值

**结束。**