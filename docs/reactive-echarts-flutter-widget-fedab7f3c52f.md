# 反应式 Echarts 颤振小工具

> 原文：<https://medium.com/analytics-vidhya/reactive-echarts-flutter-widget-fedab7f3c52f?source=collection_archive---------9----------------------->

![](img/7c9637be285c6fdabb8557bc4c344bf2.png)

> *介绍一个 reactive Echarts Flutter Widget 的开发工作:*[*Flutter _ Echarts*](https://github.com/entronad/flutter_echarts)*。*
> 
> [*知识库*](https://github.com/entronad/flutter_echarts)
> 
> [*发布开发*](https://pub.dev/packages/flutter_echarts)

随着其快速发展，Flutter 已经被应用到越来越多的大型项目中，复杂的数据可视化图表已经成为一个重要的需求。尽管 Flutter 有强大的类如 Painter 或 Canvas 来完成绘制工作，但不幸的是，在 Flutter 生态系统中仍然没有杀手级的数据可视化库。

今年年初，Flutter 开发团队发布了一个官方的内嵌 WebView 小部件: [webview_flutter](https://pub.dev/packages/webview_flutter) 。它基于新的平台视图，这使得在 Flutter 中无缝嵌入 web 内容成为可能，就像其他小部件一样。因此，我们可以将那些成熟的 web 数据可视化库导入到我们的 Flutter 应用程序中。

当谈到成熟、强大和易于使用的数据可视化库时， [Echarts](https://www.echartsjs.com/zh/index.html) 无疑是一个不错的选择。这里我就不再重复它的优点了。如果我们能够在我们的 Flutter 应用程序中添加 Echarts，我们不仅可以实现它支持的丰富的图表类型，还可以重用 web 的现成图表代码来减少工作量。

于是我们封装了一个 Flutter widget:[Flutter _ echarts](https://github.com/entronad/flutter_echarts)，兼顾扩展性和易用性，帮助 Flutter 开发者充分发挥 e charts 的功能。

# 特征

在此之前，我们在 React Native 中封装了一个 [Echarts 组件](https://github.com/entronad/react-native-echarts-demo)，获得了一些关于如何在一个反应式 UI 框架中使用数据可视化库的经验，所以在谈到 Flutter 时，我们为 flutter_echarts 设计了一些特性:

**无功更新**

Flutter 最重要的一个特点就是和其他所有反应式 UI 框架一样，根据数据的变化自动更新视图，给开发带来了很多便利。Echarts 独立于任何 UI 框架，但它是由数据驱动设计的，数据的变化驱动图表的变化。

所以我们只需要把 Echarts 的数据驱动方法和 Flutter 的视图更新连接起来，就可以实现 widget 的反应式更新。在 Echarts 中设置动态数据更新非常简单。所有数据更新都是通过`setOption`。你只需要随心所欲的获取数据，将数据填入`setOption`就可以了，不用考虑数据带来的变化，ECharts 会找出两组数据之间的差异，并通过适当的动画呈现出来。

同时，在 Flutter 中，当容器 widget 更新，传递给子 widget 的数据属性发生变化时，这个 StatefulWidget 的`State.didUpdateWidget`就会被触发。因此调用 it 中的`setOption`将通知 Echarts 更新图表。这使得 flutter_echarts 像一个简单的无状态小部件一样易于使用。

**双向通信**

图表和外部程序之间的通信是非常必要的。在 flutter_echarts 中，JavaScript 和 Dart 的通信原理就像父子小部件一样:“道具向下，事件向上”。

所有来自外部的设置和命令都通过`option`和`extraScript`以 JavaScript 代码串的形式传递给 chart。这些代码将由 WebView 执行；另一方面，WebView 内部的事件通过 JavascriptChannel 发送，并由 onMessage 函数处理。这是内部 JavaScript 和外部 Dart 之间的双向通信。

**可配置扩展**

Echarts 有各种[扩展](https://echarts.apache.org/en/download-extension.html)，包括图表、地图和 WebGL。在 web 上，我们可以将它们作为脚本文件导入，以扩展 Echarts 的功能。为了开箱即用，flutter_echarts 嵌入了最新版本的 echarts 脚本，无需额外导入。同时，我们为用户公开了一个`extensions`属性，以包含任何需要的脚本。`extensions`是字符串列表，用户可以直接将脚本字符串复制到源代码中，避免了文件读取和复杂的资产目录。

# 小部件属性

在封装小部件时，易用性通常比完整性更重要。它应该能够让所有级别的开发人员开箱即用。Echarts it self 的设计以易用性为原则，尽量把所有配置都放在`option` ( [细节在本文](http://www.cad.zju.edu.cn/home/vagblog/VAG_Work/echarts.pdf))。所以 flutter_echarts 也简化了小部件属性:

**选项**

*字符串*

字符串形式的图表的 JavaScript Echarts 选项。echarts 主要由该属性配置。您可以使用 dart:convert 中的`jsonEncode()`函数来转换 dart 对象形式的数据:

```
source: ${jsonEncode(_data1)},
```

因为 JavaScript 没有`'''`，所以可以使用这个操作符来减少配额的一些转义符:

```
Echarts(
  option: '''

    // option string

  ''',
),
```

**脚本外**

*字符串*

将在`Echarts.init()`之后和任何`chart.setOption()`之前执行的 JavaScript。这个小部件已经构建了一个名为`Messager`的 javascriptChennel，所以您可以使用这个标识符从 JavaScript 向 Flutter 发送消息:

```
extraScript: '''
  chart.on('click', (params) => {
  if(params.componentType === 'series') {
    Messager.postMessage('anything');
  }
  });
''',
```

**on 消息**

*void 函数(字符串)*

处理`extraScript`中`Messager.postMessage()`发送的消息的功能。

**扩展**

*列表<字符串>*

由 Echarts 扩展产生的字符串列表，如组件、WebGL、语言等。你可以在这里下载它们。将它们作为原始字符串插入:

```
const liquidPlugin = r'''
​
  // copy from liquid.min.js
​
''';
```

这是 flutter_echarts 的所有 4 个属性。诸如何时更新图表之类的其他事情是由内部机制决定的。这使得 flutter_echarts 看起来就像一个简单的表示性无状态小部件。用户只需要熟悉电子海图，不需要额外的学习。

完整的例子在这里:[flutter _ e charts _ example](https://github.com/entronad/flutter_echarts/tree/master/example)。

当然，如果你有任何建议或要求，请提出。

# 源代码分析

**加载 html**

对于跨平台开发，由于操作系统的文件系统不同，管理资产目录总是有困难。在 React Native 中，有时你甚至需要手动将 html 文件复制到 Android 目录中。Flutter 有一个完整的资产系统，但是它也需要额外的依赖和配置。因此，在源代码中加载本地 htmls 作为文本字符串是一个好主意，webview_flutter 团队也在其“官方示例”中推荐了这种方式。

因此，我们在小部件的初始化中将所有模板 html、Echarts 脚本、扩展脚本和初始代码放入一个字符串中，并将其作为 uri 源加载:

```
@override
  void initState() {
    super.initState();
    _htmlBase64 = 'data:text/html;base64,' + base64Encode(
      const Utf8Encoder().convert(_getHtml(
        echartsScript,
        widget.extensions ?? [],
        widget.extraScript ?? '',
      ))
    );
    _currentOption = widget.option;
  }

  ...

  @override
  Widget build(BuildContext context) {
    return WebView(
      initialUrl: _htmlBase64,

      ...
    );
  }
```

注意，在 uri 字符串中，有一些有限的字符，所以我们将字符串编码为 Base64。

有一个提示:JavaScript 没有`'''`，所以我们可以用它包装我们的 JavaScript 字符串，以减少一些逃避的工作。

**更新图表**

反应式更新的基本机制是调用 State.didUpdateWidget 钩子中的`setOption`来通知图表更新:

```
void update(String preOption) async {
    _currentOption = widget.option;
    if (_currentOption != preOption) {
      await _controller?.evaluateJavascript('''
        chart && chart.setOption($_currentOption, true);
      ''');
    }
  }
​
  @override
  void didUpdateWidget(Echarts oldWidget) {
    super.didUpdateWidget(oldWidget);
    update(oldWidget.option);
  }
```

最麻烦的部分是小部件的初始化。

我们知道 WebView 中 html 的加载和数据的获取都是异步的，我们不知道哪一个会更早完成。WebView 初始化的生命周期顺序是:

```
onWebViewCreated --> loading html --> onPageFinished
```

而 WebViewController 只能在 onWebViewCreated 中访问。换句话说，当 widgetd 获得一个 WebViewController 时，我们无法判断 html 是否已经被加载，所以在`didUpdateWidget`中，我们无法通过测试 WebViewController 来判断它是否准备好更新。

我们的解决方案是将“数据属性改变触发图表更新”分解为两个步骤:“数据属性改变导致 _currentOption 改变”和“根据 _currentOption 更新图表”，这确保任何数据都被记录，甚至在加载 html 之前。

```
String _currentOption;

  void init() async {
    await _controller?.evaluateJavascript('''
      chart.setOption($_currentOption, true);
    ''');
  }
​
  void update(String preOption) async {
    _currentOption = widget.option;
    ...
  }

  @override
  Widget build(BuildContext context) {
    return WebView(
      ...
      onPageFinished: (String url) {
        init();
      },
      ...
    );
  }
```

**消息通道**

webview_flutter 提供了一个 javascriptChannels 属性来设置多个命名通道。但是考虑到不了解 webview_flutter 的用户，flutter_echarts 并没有暴露这个属性。相反，我们只构建了一个名为“Messager”的通道:

```
@override
  Widget build(BuildContext context) {
    return WebView(
      ...
      javascriptChannels: <JavascriptChannel>[
        JavascriptChannel(
          name: 'Messager',
          onMessageReceived: (JavascriptMessage javascriptMessage) {
            widget?.onMessage(javascriptMessage.message);
          }
        ),
      ].toSet(),
    );
  }
```

如果需要发送多种类型的事件，用户可以创建类似 redux:

```
chart.on('click', (params) => {
  if(params.componentType === 'series') {
    Messager.postMessage(JSON.stringify({
      type: 'select',
      payload: params.dataIndex,
    }));
  }
});
```