# 进程、模块和混合:Ruby 的冷板凳

> 原文：<https://medium.com/analytics-vidhya/procs-modules-and-mixins-rubys-benchwarmers-12bfe4835114?source=collection_archive---------11----------------------->

![](img/3b85dc2d816e10226a5c210c3e5d7ecd.png)

**简介**

在许多情况下，基本的 OOP 教育关注于数据类型、变量、方法、数组和散列以及类。有很多很好的理由从基础开始，并在深入探索较少导航、动荡的水域之前熟悉这些工具。事实上，一些最古老的人类智慧赞美简单。然而，在对基础知识达到一定程度的熟悉后，程序员应该向前看，以便继续成长。因此，基础成为允许程序员使用其他精心设计的库和框架的构建块。一般来说，一个人对任何主题的基础理解得越好，新的和复杂的主题就越容易理解。当然，这篇文章的目的并不是建议程序员新手应该学习更少的基本技能；相反，它们是编程不可或缺的一部分。

然而，就其本身而言，以其最基本的形式，它们并没有涵盖软件工程师在该领域中面临的所有情况——呃，在计算机屏幕上。因此，如果基础如此之好，但它们没有提供足够的功能来涵盖每一个设计实现，那么 pray tell 是一个编码器要做什么呢？你这么问很有趣！在软件应用程序中创建对象、函数和类时，有些用例需要更大的灵活性。在这篇文章中，我想探讨 OOPs 的一些缺点，然后对 Ruby 提供给我们的一些工具做一个广泛而基本的介绍，这些工具增强了 OOP 最基本的构件的默认功能:过程、模块和混合。

**让我们探讨一下 OOP 的一些缺点**

面向对象编程经常被标榜为解决问题能力和工程能力的表面典范。然而，在面向对象编程之前，在计算机科学领域还有其他的范例，例如过程化或函数式编程。事实上，在一些问题上，过程式编程比面向对象编程更有效。因此，如果至少有两种关于*如何*编程的思想流派，并且仅仅是这两种流派的存在就表明它们是为解决一组特定的问题而开发的，那么有理由认为至少在它们之间存在一些折衷，并且由此引申出 OOP 的缺点。

OOP 的缺点是概念上的、风格上的和架构上的。众所周知，面向对象编程可能会令人困惑，因此有一个陡峭的学习曲线。这需要时间来适应如何用这种方法解决问题，因为*如何创建*对象以及其他细节一开始可能看起来不符合逻辑。此外，面向对象的设计可能很冗长。让我吃惊的是，一个符合惯例的程序需要这么多的包、文件和目录来完成相对简单的任务。另一种思考方式是开销。作为程序员，程序经常会变得比我们最初预期的还要大。因此，程序的效率和性能会因其自身的大小和架构而受到影响。甚至，继承 OOPs 定义特性之一——也有它的缺点，在某些边缘情况下，一个类家族的成员具有或缺乏某些与其自己的层次结构共有的属性，但是它与另一个不相关的类共享。

幸运的是，有一些策略和特性可以用来减少程序的大小，减少样板代码的重复，以及容器化组件逻辑。当然，当开始永无止境的编程之旅时，这些细节可能看起来微不足道。更愤世嫉俗的是，这些担忧可能看起来很浮夸，因为它们只是炫耀的诡计，使用了比必要更复杂的语法；当一个问题可以用简单明了的代码通过一点额外的工作来解决时，却隐藏了清晰的含义。但是在踏上黄砖路的过程中，我们可以学到很多东西，哪怕只是为了最终揭露一个假巫师。

**Procs**

proc 是封装块的对象。很简单，下课。JK。为了理解这句话的力量，我们有必要多花一点时间来理解它。首先，文字代码块——do 和 end 之间或{ and }之间的代码——直到现在都不可重用，也不可传递。因此，通过封装，我们可以抽象出一些重复，并利用块级别的作用域。因此，过程增加了代码的模块性，减少了重复。具有讽刺意味的是，Ruby 文档吹捧 procs 是“Ruby 中的一个基本概念，也是其函数式编程特性的核心”。也许将它定义为 Ruby 函数式编程特性的核心，可以帮助我们理解 procs 如何解决 OOPs 的一些缺点。

可以通过多种方式创建 Proc，但在本文中，我们将重点关注 Proc 类构造函数语法。这段代码类似于其他构造函数，但是注意花括号的使用会产生 Proc 数据类型，如下所示:

```
x = Proc.new {}puts x.inspect # => #<Proc:[0x00007ff5a5900950@proc.rb](mailto:0x00007ff5a5900950@proc.rb):3>
```

好吧，那很好，很好。但是我们如何使用它们呢？嗯，使用 procs 的一种方法是替换小的、重复的代码块。例如，如果您在程序中有一个对象，并且发现自己必须遍历该对象才能执行重复的任务，那么您可以使用 proc 来使您的代码更加 D-R-Y:

```
x = Proc.new {|num| puts num * 4}[1, 2, 3, 4, 5, 6, 7].each &x # => 4, 8, 12, 16, 20, 24, 28
```

在本例中，我们的重复任务是将数组中的数字乘以 4。请注意&符号的使用，它让 Ruby 知道“x”不是任何 ole 变量，它是一个特殊的变量。下面是另一个例子:

```
x = Proc.new { |word| word.length > 6 && word.length % 2 == 0 }long_words = ['apple', 'wisdom', 'anonymous', 'elementary', 'monitor', 'computer', 'available', 'independence']puts long_words.select &x # => elementary, computer, independence
```

在前面的示例中，proc 用于替换。*每种*方法。此外，您还可以将过程作为一等公民传递到方法中，并使用它们的。*谓*法:

```
x = Proc.new do |word| word.capitalize! word.reverse! puts word * 4 enddef some_method(proc_x) proc_x.call("hip hip")endsome_method(x) # => pih piHpih piHpih piHpih piH
```

此外，您可以创建具有自定义块功能的自定义方法，如下所述:

```
x = Proc.new { puts "This is Proc-tically impossible"}def some_method yieldendsome_method(&x) # => This is Proc-tically impossible
```

如果你觉得危险，procs 甚至可以用来简化一些数据类型转换和迭代器逻辑。根据 Ruby 文档，这是因为“任何实现。 *to_proc* 方法可以通过'&'操作符转换成 proc，因此可以被迭代器使用在 Ruby 的核心类中，符号、方法和散列实现了。 *to_proc* 方法。

```
arr = ["1", "2", "3"]puts arr.map &:to_i #=> 1, 2, 3 arr = ["Danger's", "my", "middle", "name"]puts arr.map &:upcase #=> DANGER'S, MY, MIDDLE, NAME
```

Procs 可用于简化块级代码和利用局部范围的变量。更重要的是，procs 是 Ruby 对闭包*的实现，闭包*允许它们记住创建它们的上下文。但那是另一天的话题。因为 proc 在功能上是一个对象，所以块本身可以在程序中传递，而不必多次重复代码。因此，我们可以将代码分配给变量，并根据需要重用、转换和插入代码。在我看来，这绝对是一个值得熟悉并保存在工具箱中的工具。

**模块和混合模块**

模块是向类提供额外状态和行为的好方法，而不必担心继承链和覆盖其他方法。它们提供对模块声明中定义的方法和常数的访问。模块提供对实例级方法和模块级方法的支持，但是*包含*或使用模块行为的类只能访问实例级方法。相反，模块级方法可以直接在模块对象上访问。

虽然面向对象继承的好处已经有了很好的证明，但是也应该注意到，由于继承链的垂直性，它有一些限制。事实上，类从它们的祖先类继承，并在它们的子类中复制功能，但是没有与不相关的类共享状态和行为的本地方法。从表面上看，在不相关的类之间共享属性似乎不合逻辑，但这在现实生活中实际上很常见。想象一下，狗和猫会引起过敏，但是花和草，花生和小麦也会引起过敏。以这种方式包装行为并在不同的类之间共享是模块的常见用法。

让我们在编码示例中尝试将一些模块方法容器化。首先，让我们创建一个父类:

```
# superhero.rbclass Superhero def righteous_call(evil_doer) puts "Hey, put that back, #{evil_doer}!" endend
```

太好了，我们降落在超级英雄身上。接下来，让我们装备他们中的一些人，制作一个只有一些超级英雄才能使用的斯塔克工业模块:

```
# stark.rbmodule Stark def thrusters puts "Boomm!!" end def ai_sunglasses(name) puts "A.I. System will track #{name} through sunglasses..." endend
```

现在，是时候制造这些超级英雄，并展示模块如何提供对基于类的继承之外的行为的访问。

```
# hero.rbrequire_relative "./superhero"
require_relative "./stark" class Spiderman < Superhero include Starkendclass Ironman < Superhero include Starkendspidey = Spiderman.new iron = Ironman.newspidey.thrusters # => Boomm!!iron.thrusters # => Boomm!!spidey.ai_sunglasses("Wilson Fisk") # => A.I. System will track Wilson Fisk through sunglasses...iron.ai_sunglasses("Victor Von Doom") # => A.I. System will track Victor Von Doom through sunglasses...
```

与类继承语法不同，类< ParentClass, modules are imported within the body of the class using the *包括*关键字。我认为这是有意义的，因为这个类没有变成新的东西，也没有获得新的属性；相反，它通过模块导入采用了新的行为。

尽管蜘蛛侠和钢铁侠都是超级英雄，但并不是所有我们创造的超级英雄都可以使用斯塔克工业的推进器和人工智能太阳镜。例如，蝙蝠侠可能是一个超级英雄，但他来自 DC 宇宙，所以他不会接触到托尼·斯塔克的好东西。为了克服继承的限制，模块成为跨类共享行为的便捷方式。

通过使用 Mixins，这个特性可以得到进一步的发展。mixin 只是在一个类中增加了对多个模块的访问。由于 Ruby 不支持多重继承，我们可以用多个模块混合成——读 Mixin——一个子类。函数混合提供了多重继承提供给其他语言的可扩展性和模块化。因此，虽然 Ruby 没有提供其他语言实现的所有 OOP 特性，但是 mixins 实现了这个功能，并且是 Ruby 中 *ducktyping* 的经典例子。根据作者 Harmes 和 Diaz 的说法， *ducktyping* 来源于这样一个表达，“如果它走路像鸭子，叫声像鸭子，那么它一定是一只鸭子。”在编程的上下文中，它可以被解释为如果一个给定的对象像另一个对象一样定义和运行，那么它也必须是第二个对象。

模块的另一个用例是为公共类名创建名称空间，避免程序中的冲突。想象一个由英雄和恶棍组成的程序。如果程序足够大，有理由认为类名冲突会导致冲突。请注意下面的示例:

首先让我们创建一个超级英雄模块:

```
module Superhero class Cyclops def eye puts "My eye(s) shoots lasers. Zap!" end end class Titans def original_team puts "Robin, Superboy, Kid Flash, Wonder Girl" end def teen_titans puts "Nightwing, Raven, Starfire, Beast Boy" end endend
```

接下来，让我们制作一个希腊语模块:

```
module Greek class Cyclops def eye puts "I traded my eye to see the future" end end class Titans def original_team puts "Oceanus, Tethys, Hyperion, Theia, Coeus, Phoebe, Cronus, Rhea, Mnemosyne, Themis, Crius and Iapetus" end endend
```

现在，是时候证明通过使用模块，Titans 和 Cyclops 类的两个版本可以存在于同一个 Ruby 文件中，而不会有任何命名空间错误。

```
require_relative './superhero'
require_relative './greek' s_cyclops = Superhero::Cyclops.newg_cyclops = Greek::Cyclops.news_titans = Superhero::Titans.newg_titans = Greek::Titans.news_cyclops.eye # => My eye(s) shoots lasers. Zap!g_cyclops.eye # => I traded my eye to see the futures_titans.original_team # => Robin, Superboy, Kid Flash, Wonder Girlg_titans.original_team # => Oceanus, Tethys, Hyperion, Theia, Coeus, Phoebe, Cronus, Rhea, Mnemosyne, Themis, Crius and Iapetus
```

在继承和模块之间，Ruby 为软件工程师提供了解决几乎所有类问题的工具。继承是 OOP 的原则之一，但是尽管它提供了强大的功能和多样性，它并不能解决所有的设计挑战。由于它们与这一组具有挑战性的边缘案例相关，模块提供了模块化和更高的灵活性，同时减少了重复。尽管模块不是*可实例化的*，但是它们确实提供了一个干净地组织、定义和构造代码的名称空间。区分何时使用基于类的继承和混合模块的简单方法是定义关系。蜘蛛侠”是一个“超级英雄；而他只有一个斯塔克工业推进器和人工智能太阳镜。

**结论**

Ruby 为程序员提供了面向对象语言中所有的基本功能。然后，Ruby 通过它的一些高级特性，如进程、模块和混合，增加了灵活性、可扩展性和模块化。这些特性通过提供额外的工具来增强程序员坚持设计理念的能力，从而完善了 Ruby 的使用。

**参考书目**

*   [https://ruby-doc.org/core-2.6/Proc.html](https://ruby-doc.org/core-2.6/Proc.html)
*   [https://ruby-doc.org/core-2.6/Module.html](https://ruby-doc.org/core-2.6/Module.html)
*   [https://resources . say lor . org/www resources/archived/site/WP-content/uploads/2013/02/cs 101-2 . 1 . 2-advantages ofoop-final . pdf](https://resources.saylor.org/wwwresources/archived/site/wp-content/uploads/2013/02/CS101-2.1.2-AdvantagesDisadvantagesOfOOP-FINAL.pdf)
*   [https://launchschool.com/books/oo_ruby/read/inheritance](https://launchschool.com/books/oo_ruby/read/inheritance)
*   [http://rubylearning.com/satishtalim/ruby_inheritance.html](http://rubylearning.com/satishtalim/ruby_inheritance.html)