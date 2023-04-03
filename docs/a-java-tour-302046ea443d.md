# 爪哇之旅

> 原文：<https://medium.com/analytics-vidhya/a-java-tour-302046ea443d?source=collection_archive---------24----------------------->

让我们利用这些天的空闲时间重温 Java 编程语言吧！

![](img/245e0dfb2c50468589f6c7c716f57888.png)

Java 被广泛认为是产生高质量代码的强大语言……至少如果你能驾驭野兽的话！因为 Java 也确实是一种(相当)冗长的语言，它不会试图隐藏任何东西:缺乏语法上的好处可能会让您感觉更接近计算机实际做的事情，这一特性可能会让通常使用 JavaScript 或 Python 的初学者甚至高级脚本编写人员感到沮丧。尽管如此，Java 社区很清楚这一点，并努力以完全向后兼容的方式改进产品*。*

也许你学过 Java 编程，也许是彻底的灾难！然后和很多人一样，你发现了 Python，TypeScript，Kotline 什么的，感觉轻而易举！这些年来，你一直认为 Java 是一种过时的语言，运行缓慢，还在使用，只是因为现有的代码库。好吧，让我们看看现代 Java 到底是什么样子。

在这次闲聊中，我们将研究一些现实问题的一个用例。客户有一个数据源(数据库、文件等)，我们可以通过查询来获取表单信息

```
people_name ; zip_code ; waste_weight_yesterday
```

我们不能像往常一样直接访问数据源。相反，我们需要向外部服务查询它:

```
public interface RemoteService {
   InputStream datasource() throws IOException;
}
```

客户要求我们根据邮政编码(`String`)找出昨天产生最多垃圾的前 3 名(`String`)。数据源应该没有任何损坏的数据(这通常不会发生，但让我们假设一下！).客户要求用 Java 来完成，以便集成到自己的代码库中。但愿，这是个好机会！我写的就是这个问题，所以你不用写任何代码，只需复制/粘贴:你真幸运。

一个好的架构的最低要求是至少有一个描述我们计算结果的接口草图。基本上，我们的报告将是一些我们可以如下操作的对象:

```
interface Report { enum TopPosition {
      TOP_1, TOP_2, TOP_3;
   } Optional<Entry> getTop(String zipCode, TopPosition position);
   List<String> getAllZipCodes();
}record Entry {
   String name();
   String zipCode();
   int wasteAmount();
}
```

*注意:*我们稍后将讨论记录类型。现在，让我们跳过这个！

*注意:*我们本应该用`Set`而不是`List`来设计界面，但是它将不会很好地适合下面的段落！所以过了这一段，自己做把`List`改成`Set`的心理练习吧！

在实现任何进一步的机制之前，让我们以一种非平凡的方式通过查询`Report`来热身。我们要做的小练习是**找到所有前 1 名**。让我们看看不同的实现。首先是糟糕的 Java:

```
List<Entry> top1s = new ArrayList<>(); // array-like, with mut. size
List<String> codes = report.getAllZipCodes();
for (int i = 0; i < codes.size(); i++) {
   Optional<Entry> top1 = getTop(codes.get(i), TOP_1);
   if (top1.isPresent()) {
      top1s.add(top1.get());
   }
}
```

不用说，这个实现(遗憾的是它仍然存在于社区中)是一个明智的宣战。如果你的`List`是一个`LinkedList`，上面的这个迭代很大程度上是为了`O(n²)`。此外，`ArrayList`实现可能不是我们所做的许多推送的最佳选择，因为内部数组缓冲区需要被复制很多次。不幸的是，学生们经常被告知要这样编码。

让我们看看另一种更现代的实现方式:

```
List<Entry> top1s = new Stack<>(); // pushing in stack is O(1)
for (String zipCode : report.getAllZipCodes())
   getTop(zipCode, TOP_1).ifPresent(top1s::push);
```

在上面，我们强调了这样一个事实，方法引用`top1s::push`可能被认为是 lambda 表达式，而这又可能被认为是匿名接口实例化。这里的接口是 Java 猜测的，因为`Optional<Entry>`上的`ifPresent`只需要一个`Consumer<Entry>`，所以

```
Consumer<Entry> c = top1s::push;
```

在语法上等同于

```
Consumer<Entry> c = entry -> top1s.push(entry);
```

这在语法上相当于

```
Consumer<Entry> c = new Consumer<>() {
   @Override public void accept(Entry entry) {
      top1s.push(entry);
   }
}
```

lambda 表达式和方法引用可能被认为是后者的语法糖。其实也有编制差异。一般来说，与匿名类方法相比，编译器更喜欢方法引用方法。对于 lambda 表达式，这实际上取决于 lambda 的内容。

我们可以通过避免先验的 boiler-plated 空列表初始化来进一步研究上面的例子:

```
List<Entry> top1s = report.getAllZipCodes()
       .stream()
       .map(zipCode -> report.getTop(zipCode, TOP_1))
       .flatMap(Optional::stream)
       .collect(Collectors.toList());
```

有趣的是，虽然是以函数式风格编写的，但后者实际上比第二种基于循环的方法更冗长，而且没有太多的表现力。我们也失去了显式选择列表实现的好处。

现在让我们来看一下报告的实施和数据处理本身。我们将一步一步来。首先，我们将尽快处理这些行以释放资源:

```
Report process (RemoteService service) throws IOException { List<String> rows = new Stack<>(); try(InputStream inputStream = service.datasource();
       Scanner scanner = new Scanner(inputStream)
   ) {
      while(scanner.hasNextLine())
         rows.push(scanner.nextLine());
   }
```

也许你不熟悉 try-with-resource 语法，它也是 JDK7 以来的一个语言特性。try-with-resource 参数可以接受任何列表中的`AutoCloseable`子类型，并且在 try-block 结束时，或者如果发生任何异常，每个提供的`AutoCloseable` with be…会自动关闭。这意味着在 Java 中，永远不要自己调用`close`方法。这类似于 Python 的`with`语法。

我们现在要处理每一行。客户需要我们根据邮政编码汇总每个`String`，然后找到前 3 个。让我们将每个`String`包装到一些更方便的类中:

```
class Row {
   private final String source, zipCode;
   private final int wasteAmount;
   Row(String source) {
      this.source = source;
      zipCode = source.substring(
                        source.indexOf(";")+1,
                        source.lastIndexOf(";")
                    );
      wasteAmount = Integer.parseInt(
                        source.substring(source.lastIndexOf(";"))
                    );
   } String getZipCode() { return zipCode; }
   String getName() {
      return source.substring(0, source.indexOf(";"));
   }
   int getWasteAmount() { return wasteAmount; }
}
```

正如您所看到的，这个类是从一个`String`构造的，但是保持了一个更复杂的内部状态，因为它预先提取了邮政编码，并且将垃圾量转换为整数。最后，它提供了以更透明的方式访问这些信息和名称的方法。还要注意完全保护实例内部状态的封装模式。从外部的角度来看，只有 getters 存在:这个类的客户不应该知道更多。

这样包装`String`源有效率吗？`String`的开销大约是内部数组`char[]`引用的 64 位+数组中每个字符的 16 位 x 长度，所以

```
String_weight ~ 64 + length * 16
```

在一个常见的用例中，姓名的顺序是 2⁴，邮政编码的顺序是 2。浪费量用小数表示应该挺小的，再说 2 位数吧。因此`source`场的权重大约是

```
source_weight ~ 64 + 2^5 * 16 ~ 2^9 
```

通过内部对象引用，我们的类的权重是 64 位(我们有 2 个，2 个`String`)，32 位用于 int 原语类型，以及每个字符串权重的额外成本，所以:

```
row_weight ~ 64 + (64 + 2^5 * 16) + 64 + (64 + 2^2 * 16) + 32 ~ 2^10
```

比例是 1.5，没那么大。从`String`到`Row`的转换将不再是一个问题，它将极大地改善算法方面。使用`Stream`方式可以非常平稳地完成转换:

```
rows.stream().map(Row::new)  // Stream<Row>
```

看看我们如何通过方法引用调用`Row`的构造函数。我们知道有一连串的行。没有必要再次将其转换为`List`，我们已经可以使用流的`collect`工具进行聚合:

```
rows.stream().map(Row::new)
             .collect(Collectors.groupingBy(Row::getZip))
```

分组的结果是一个将每个邮政编码(`String`)映射到相应的`Row`列表的`Map<String,List<Row>>`。我们现在将迭代这个图的每个键，对于每个键，根据浪费量对列表进行排序，并将结果限制为 3 个元素。同样，`Stream`接口足够丰富，可以处理这种计算:

```
rows.stream().map(Row::new)
             .collect(Collectors.groupingBy(Row::getZip))
             .entrySet().stream()
             // Stream<Map.Entry<String, List<Row>>>
             .collect(Collectors.toMap(
                 Map.Entry::getKey,
                 entry -> entry.getValue().stream() // Stream<Row>
                            .sorted(???)            // to do
                            .limit(3)               // Stream<Row>
                            .toArray(Row[]::new)    // Row[]
             ));
```

在上面这里，蒸汽上的`sorted`方法需要一个`Comparator<Row>`。定义非常清楚:

```
@FunctionalInterface public interface Comparator<T>
```

`@FunctionalInterface`注释意味着一个比较器只有一个抽象方法。这表明有人被邀请在类实现中或通过方法引用实例化这样的接口。

在我们的例子中，在`Row`的层次上实现`Comparator`接口没有多大意义，因为从业务的角度来看，它们并不都是可比较的:我们只是被要求逐个邮政编码地比较它们。(当然，对它们进行全球比较是有意义的，但这是一种推断)。深入研究一下，我们发现 JDK8 中有一个工具可以从一个`Row`中提取 int，并将其用作比较键:

```
.sorted(Comparator.comparingInt(Row::getWasteAmount))
```

总而言之，我们现在已经

```
var mapping
  = rows.stream().map(Row::new)
        .collect(Collectors.groupingBy(Row::getZip))
        .entrySet().stream()
        .collect(Collectors.toMap(Map.Entry::getKey,
            entry -> entry.getValue().stream()
                          .sorted(comparingInt(row::getWasteAmount))
                          .limit(3)
                          .toArray(Row[]::new)
        ));
```

在上面的最后一次收集中，我们可以使用`List`进行收集，但是数组解决方案通常更便宜。

我们还使用了特殊的`var`关键字，因为编译器可以自己猜测正确的成员是一个`Map<String,Row[]>`，没有必要重复。`var`是 JDK10 的一个特性。

所有的线都处理成功了，干得好！现在让我们实现我们的`Report`和`Entry`类型。我们已经在一个`Map<String, Row[]>`中收集了所有的信息，并且我们已经完全控制了地图，因为我们是它的创造者。让我们将这些信息封装在一个`Report`外壳中:

```
class ReportImpl implements Report {
   private final Map<String, Row[]> mapping;
   ReportImpl(Map<String, Row[]> mapping) {
      this.mapping = mapping;
   } @Override public Set<? extends String> getAllZipCodes() {
      return mapping.keySet();
   } @Override public Optional<Entry> getTop(String zipCode,
                                           TopPosition position) {
      var idx = switch(position) {
         case TOP_1 -> 0;
         case TOP_2 -> 1;
         case TOP_3 -> { yield 2; };
      }
      return Optional.ofNullable(mapping.get(zipCode))
                     .map(arr -> arr.length > idx ? arr[idx]: null)
                     .map(???);
   }
}
```

注意使用了`var`关键字来声明一个由右成员初始化的变量。这里，编译器可以自己推断出`idx`的类型是`int`。

还要注意上面开关表达式的使用。在 JDK14 中，开关表达式允许在基本类型、字符串和枚举上进行模式匹配(实际上是通常的开关)，有以下区别:

1.  没有失败行为:不需要在案例之间中断，这更像是一个映射
2.  映射的右侧可以是一个以`yield`结束的块，以提供一个值，就像我们的第三个例子
3.  在枚举匹配的情况下，不需要缺省情况:编译器检测映射是否穷尽。

JDK15 中的`sealed class`旨在改进开关表达式，但那是以后的事了！

对于我们来说，我们现在需要的是能够创建一个`Entry`。因为一个`Entry`实际上是暴露给客户端的最后一个流程步骤，所以我们不会让它成为一个`Entry`来污染我们的`Row`类，尽管它已经包含了所有的信息！为什么？因为我们希望将这两者分开:`Row`旨在表示一个我们可以计算的可处理实体，而`Entry`只是一个数据传输对象，就像数据的快照。

这非常适合 JDK14 中作为预览功能引入的`record`!记录只不过是一个类，除了在其构造函数中提供的字段之外，不能包含任何其他字段。还记得我们的`Row`类是怎么回事吗？那是因为一个`Row`确实是我们操纵的东西，它可能包含很多信息。它显然不同于其构造函数参数的总和。对于一个更像是我们想要与客户端共享的一些异构元组的`Entry`,情况可能会有所不同。

```
record Entry(String name,String zipCode,int wasteAmount) {
   EntryDto { // post-construct process, the record component exist!
      assert name != null;
      assert zipCode != null;
   }
}
```

记录机制为构造函数中提供的每个`x`字段(组件)自动生成一个方法`x()`。它还接受一个后构造函数块，当提供组件时将调用该块。我们可以使用它进行非常基本的验证(这里我们防止空引用)。

记录机制还会根据字段自动为我们生成`hashCode`和`equals`方法。`record`是一种 Scala case 类，区别在于:

1.  除了在构建时提供的字段之外，它不能包含其他字段
2.  无论是类还是记录，都不能从另一个记录继承

以下是最终的代码:

```
record Entry(String name,String zipCode,int wasteAmount) {
   Entry {
      assert name != null;
      assert zipCode != null;
   }
}class Row {
   private final String source, zipCode;
   private final int wasteAmount;
   Row(String source) {
      this.source = source;
      zipCode = source.substring(
                        source.indexOf(";")+1,
                        source.lastIndexOf(";")
                    ).trim();
      wasteAmount = Integer.parseInt(
                        source.substring(source.lastIndexOf(";"))
                              .trim()
                    );
   } String getZipCode() { return zipCode; }
   String getName() {
      return source.substring(0, source.indexOf(";")).trim();
   }
   int getWasteAmount() { return wasteAmount; }
}class ReportImpl implements Report {
   private final Map<String, Row[]> mapping;
   ReportImpl(Map<String, Row[]> mapping) {
      this.mapping = mapping;
   } @Override public Set<? extends String> getAllZipCodes() {
      return mapping.keySet();
   }@Override public Optional<Entry> getTop(String zipCode,
                                        TopPosition position) {
      var idx = switch(position) {
         case TOP_1 -> 0;
         case TOP_2 -> 1;
         case TOP_3 -> 2;
      }
      return Optional.ofNullable(mapping.get(zipCode))
                     .map(arr -> arr.length > idx ? arr[idx]: null)
                     .map(row -> new Entry(
                                        row.getName(),
                                        row.getZipCode(),
                                        row.getWasteAmount()
                     ));
   }
}Report process (RemoteService service) throws IOException {
   var rows = new Stack<String>();
   try(InputStream inputStream = service.datasource();
       Scanner scanner = new Scanner(inputStream)
   ) {
      while(scanner.hasNextLine())
         rows.push(scanner.nextLine());
   }
   var mapping = rows.stream()
        .map(Row::new)
        .collect(Collectors.groupingBy(Row::getZip))
        .entrySet().stream()
        .collect(Collectors.toMap(Map.Entry::getKey,
            entry -> entry.getValue().stream()
                          .sorted(comparingInt(row::getWasteAmount))
                          .limit(3)
                          .toArray(Row[]::new)
        ));
   return new ReportImpl(mapping);
}
```

希望你在我们的聊天中学到了一些关于 Java 的知识！

我希望你现在确信 Java 不是纯粹邪恶的冗长语言。不要犹豫，分享你的感受吧！