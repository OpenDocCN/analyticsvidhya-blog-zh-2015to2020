# NodeJS 中的 OS 模块

> 原文：<https://medium.com/analytics-vidhya/os-module-in-nodejs-26459c17f5a?source=collection_archive---------13----------------------->

OS 模块提供与操作系统和硬件相关的信息。

![](img/a70bac901be2c0fa0fb676fc844653fe.png)

照片由[帕诺斯·萨卡拉基斯](https://unsplash.com/@meymigrou?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

`os`模块提供了获取硬件相关信息的 API，比如 CPU、内存、目录、IP 地址等等。

在本教程中，我们将学习`os` 模块的基本方法和一些概念。

要开始使用`os`模块，我们必须在项目中导入`os`模块。

```
const os = require('os);
```

## os.arch()

这个方法将返回处理器的架构。

```
const os **=** require('os');console**.**log(os**.**arch());
```

输出:

```
x64
```

## os.cpus()

此方法返回包含逻辑 CPU 信息的对象数组。

```
const os = require('os);console.log(os.cpus());
```

输出:

```
[
  {
    model: 'Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz',
    speed: 2496,
    times: {
      user: 6088781,
      nice: 0,
      sys: 6719718,
      idle: 44304250,
      irq: 1336078
    }
  },
  {
    model: 'Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz',
    speed: 2496,
    times: {
      user: 6955656,
      nice: 0,
      sys: 5061265,
      idle: 45095515,
      irq: 169078
    }
  },
  {
    model: 'Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz',
    speed: 2496,
    times: {
      user: 6605562,
      nice: 0,
      sys: 4668015,
      idle: 45838859,
      irq: 95781
    }
  },
  {
    model: 'Intel(R) Core(TM) i5-7300HQ CPU @ 2.50GHz',
    speed: 2496,
    times: {
      user: 7106781,
      nice: 0,
      sys: 4634140,
      idle: 45371515,
      irq: 68109
    }
  }
]
```

## os.freemem()

这个方法返回整数形式的空闲内存字节。

```
const os **=** require('os');console**.**log(os**.**freemem());
```

输出:

```
2954100736
```

## os.getPriority(pid)

此方法返回进程的调度优先级。

我们必须通过`pid`作为论据。

```
const os **=** require('os');console**.**log(os**.**getPriority(13512));
```

这是我的案例，我提供了`Windows's File Explorer`中的`pid`。

输出:

```
0
```

## os.homedir()

该方法将当前用户的主目录作为一个字符串。

```
const os **=** require('os');console**.**log(os**.**homedir());
```

输出:

```
C:\Users\Prathamesh More
```

## os.hostname()

该方法返回操作系统的主机名，即字符串形式的计算机名。

```
const os **=** require('os');console**.**log(os**.**hostname());
```

输出:

```
Prathamesh-Omen
```

## 操作系统.网络接口()

此方法返回包含网络接口设备信息的对象。

```
const os **=** require('os');console**.**log(os**.**networkInterfaces());
```

输出:

```
{
  WiFi: [
    {
      address: 'fe80::bc77:2fb3:a3f:1e22',
      netmask: 'ffff:ffff:ffff:ffff::',
      family: 'IPv6',
      mac: '88:78:73:ef:c1:e9',
      internal: false,
      cidr: 'fe80::bc77:2fb3:a3f:1e22/64',
      scopeid: 6
    },
    {
      address: '192.168.43.61',
      netmask: '255.255.255.0',
      family: 'IPv4',
      mac: '88:78:73:ef:c1:e9',
      internal: false,
      cidr: '192.168.43.61/24'
    }
  ],
  'Local Area Connection* 2': [
    {
      address: 'fe80::c0fc:bfd3:8f90:7a1c',
      netmask: 'ffff:ffff:ffff:ffff::',
      family: 'IPv6',
      mac: '8a:78:73:ef:c1:e9',
      internal: false,
      cidr: 'fe80::c0fc:bfd3:8f90:7a1c/64',
      scopeid: 15
    },
    {
      address: '192.168.137.1',
      netmask: '255.255.255.0',
      family: 'IPv4',
      mac: '8a:78:73:ef:c1:e9',
      internal: false,
      cidr: '192.168.137.1/24'
    }
  ],
  'Loopback Pseudo-Interface 1': [
    {
      address: '::1',
      netmask: 'ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff',
      family: 'IPv6',
      mac: '00:00:00:00:00:00',
      internal: true,
      cidr: '::1/128',
      scopeid: 0
    },
    {
      address: '127.0.0.1',
      netmask: '255.0.0.0',
      family: 'IPv4',
      mac: '00:00:00:00:00:00',
      internal: true,
      cidr: '127.0.0.1/8'
    }
  ]
}
```

## 操作系统平台()

该方法返回平台信息，即操作系统平台，如`win64` `arm` `linux`等。

```
const os **=** require('os');console**.**log(os**.**platform());
```

输出:

```
win32
```

## os.totalmem()

此方法以字符串形式返回总系统内存(以字节为单位)。

```
const os **=** require('os');console**.**log(os**.**totalmem());
```

我有 8 场演出。

输出:

```
8469581824
```

## OS . userinfo([选项])

此方法返回当前用户。返回的对象包括`username`、`uid`、`gid`、`shell`和`homedir`。在 Windows 上，`uid`和`gid`字段是`-1`，`shell`是`null`。

选项:如果`encoding`设置为`'buffer'`，`username`，`shell`，`homedir`值将为`Buffer`实例。**默认:** `'utf8'`。

```
const os **=** require('os');console**.**log(os**.**userInfo());
```

输出:

```
{
  uid: -1,
  gid: -1,
  username: 'Prathamesh More',
  homedir: 'C:\\Users\\Prathamesh More',
  shell: null
}
```

在本教程中，我们介绍了大多数方法。你可以在 NodeJS 的[官方文档上探索更多。](https://nodejs.org/api/os.html)

谢谢大家！

编码快乐！

随时联系我！

[pprathamesmore . github . io](http://pprathameshmore.github.io/)