# 在 React 中制作日历

> 原文：<https://medium.com/analytics-vidhya/making-a-calendar-in-react-f24d583e5728?source=collection_archive---------16----------------------->

![](img/16d052d00483bdf059ab140da00e5b5b.png)

在我在 Flatiron 的最后一个项目中，我想制作一个类似于 Airbnb 的预订系统，允许用户在日历上选择一系列日期，并查看已被屏蔽的日期。经过几个小时的搜索和多次尝试，我找到了一个完美的手机友好的日期选择器库，恰巧叫做 Airbnb 的 react-dates。这个博客将是一个关于如何在你的应用程序中设置日历，划分日期和获取日期范围的教程。

1.  正在设置

首先，确保您安装了正确的依赖项。对于这个演示，我将使用 bootstrap、反应日期、矩和矩范围。

```
npm install --save bootstrap react-dates moment moment-range
```

然后转到 src>App.js 并确保您有以下导入:

```
import React, {Component} from ‘react’;
import ‘bootstrap/dist/css/bootstrap.min.css’;
import ‘react-dates/initialize’;
import ‘react-dates/lib/css/_datepicker.css’;
import {DateRangePicker} from ‘react-dates’;
import Moment from "moment";
import { extendMoment } from "moment-range";
import ‘./App.css’;
```

完成后，将以下代码复制到 App.js 中:

```
class App extends Component {
  state = {
    startDate: null,
    endDate: null
  };render() {
    return (
    <div className='App'>
      <DateRangePicker
        startDate={this.state.startDate} 
        startDateId="your_unique_start_date_id" 
        endDate={this.state.endDate} 
        endDateId="your_unique_end_date_id" 
        onDatesChange={({ startDate, endDate }) => this.setState({  startDate, endDate })} 
        focusedInput={this.state.focusedInput} 
        onFocusChange={focusedInput => this.setState({ focusedInput })}
      />
</div>
  );
 }
}export default App;
```

此时，您应该已经能够在运行 npm install 时看到日历了！就这么简单！但是我猜你和我一样，也想用这个日历做点什么。如果你看一下 [react-dates](https://github.com/airbnb/react-dates) 的文档(我在下面做了链接)，你会发现你可以为 DateRangePicker 提供很多支持。你可以浏览[故事书](http://airbnb.io/react-dates/?path=/story/drp-input-props--default)来了解所有这些，但在接下来的几节中，我将专门讲述我是如何阻止日期和抓取日期的。

2.取消约会

React-date 好心地给我们提供了一个道具，让你传递一个函数，它会相应地屏蔽掉日期——**is day blocked**。在做了一些检测工作(也就是控制台日志记录)之后，我发现**被阻塞的函数**一次给函数一个日期，当它从函数中出来时返回 true 或 false。如果返回值为真，那么这一天将被封锁。听起来很简单，对吗？！但是没有。对于我的应用程序，我需要在数据库中存储一个开始日期和一个结束日期。我意识到我需要创建一个范围，将中间的所有日期都屏蔽掉。这一部分要求你深入 moment.js 的世界，这有点吓人，但是一旦你阅读了[文档](https://www.npmjs.com/package/moment-range/v/1.0.2)，它实际上非常简单。经过大量的思考并一步一步地写出所有的东西，我想到了这个解决方案:

```
isBlocked = date => {
  let bookedRanges = [];
  let blocked; bookings.map(booking => {
    bookedRanges = [...bookedRanges, 
    moment.range(booking.startDate, booking.endDate)]
   }
  );blocked = bookedRanges.find(range => range.contains(date));return blocked;
};
```

首先，我使用 moment.range()将我的预订数组转换成一个范围数组。这需要开始日期和结束日期。然后我获取这个数组，对于 DateRangePicker 传入的每个日期，我使用。find()检查是否在我的范围内。如果是，那么它会给我一个 true 或 false 值，这个值会返回给 DateRangePicker。

3.抓住日期

我的预订日历的下一站是能够选择日期以便创建预订。为了使事情变得更简单，我使用一个带有 startDate 和 endDate 的警报来测试是否返回了正确的值。正如你在上面(和下面)看到的，startDate 和 endDate 需要有自己的状态。这非常有利于使用，因为这意味着获取数据非常容易。我制作了一个按钮，在 onClick 上为这两种状态发出警告，让自己看到状态在相应地更新。这给了我制定实际预订方法所需的一切。

这是我的 App.js 文件包含所有代码后的样子:

```
class App extends Component {
  state = {
    startDate: null,
    endDate: null
  }; alertStartDate = () => {alert(this.state.startDate)}; alertEndDate = () => {alert(this.state.endDate)}; render() {
    return (
    <div className='App'>
      <DateRangePicker
        startDate={this.state.startDate} 
        startDateId="your_unique_start_date_id" 
        endDate={this.state.endDate} 
        endDateId="your_unique_end_date_id" 
        onDatesChange={({ startDate, endDate }) => this.setState({  startDate, endDate })} 
        focusedInput={this.state.focusedInput} 
        onFocusChange={focusedInput => this.setState({ focusedInput })}
        isDayBlocked={this.isBlocked} 
      /> <button onClick={this.alertStartDate}>Click me for start date</button> <button onClick={this.alertEndDate}>Click me for end date</button> </div>
  );
 }
}export default App;
```

这是我最后一个项目中最让我紧张的部分。但是我再次被提醒，如果我把事情分解成小块，我最终会找到解决方法。即使您的应用程序不需要这种确切的功能，我也希望这篇文章是一个有用的演示，说明如何设置 react-dates 日历以及从那里开始。

来源:

[](https://github.com/airbnb/react-dates) [## Airbnb/react-date

### 一个易于国际化、易于访问、移动友好的 web 日期选择器库。例如…

github.com](https://github.com/airbnb/react-dates) 

[http://airbnb.io/react-dates/?path=/story/drp-input-props-默认](http://airbnb.io/react-dates/?path=/story/drp-input-props--default)

[](https://www.npmjs.com/package/moment-range/v/1.0.2) [## 力矩范围

### Moment.js 的奇特日期范围

www.npmjs.com](https://www.npmjs.com/package/moment-range/v/1.0.2)  [## Moment.js | Docs

### Moment 被设计为可以在浏览器和 Node.js 中工作。所有代码都应该在这两种环境中工作，并且…

momentjs.com](https://momentjs.com/docs/)