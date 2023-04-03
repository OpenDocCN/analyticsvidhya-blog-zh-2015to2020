# Python 多线程:在你的程序中引入并发性。

> 原文：<https://medium.com/analytics-vidhya/python-multithreading-introduce-concurrency-into-your-programs-bf9e837667ad?source=collection_archive---------20----------------------->

## 多线程操作

## 轻松地将任何一段 python 代码转换成异步线程程序。

![](img/09ff58fa78c3de5af2c4bfb23d0c64bb.png)

JOSHUA COLEMAN 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

> 当任务是相同的，只是它们操作的数据不同，并且可以以任何顺序执行时，问题就被说成是令人尴尬的*并行*

oncurrency 在需要执行 n 次特定动作的算法中效果最好。如果这个操作在我们的计算机范围之外(等待 api 的响应),多线程是最理想的选择，可以最大限度地减少未使用的计算资源。

Python 经常被指出难以用于多线程任务。然而, [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) 包的 ThreadPoolExecutor 模块部分(Python 标准库的一部分)为我们提供了一种简单的方法，无需对我们的原始代码进行重大修改。

# **用例:多个 api 调用**

让我们从一段简单的代码开始，这段代码将向 swapi.dev api 发出 n 个请求。SWAPI 是一个星球大战 API，它包含一个可以通过 REST API 使用的数据集。

第一个端点将给出数据集中不同人的数量:

```
**import** **requests**url = 'https://swapi.dev/api/people/'
r = requests.get(url)
print(r.json()['count'])
# >> 82
```

使用个人标识符(标识符从 1 到 83)，我们可以获得一个人的信息:

```
**import** **requests**url = 'https://swapi.dev/api/people/1'
r = requests.get(url)
print(r.json())>>
{
    'name': 'Luke Skywalker',
    'height': '172',
    'mass': '77',
    'hair_color': 'blond',
    'skin_color': 'fair',
    'eye_color': 'blue',
    'birth_year': '19BBY',
    ...
}
```

现在让我们创建一个人员 id 列表，以便能够遍历它们，向 swapi api 发出请求，并将每个 id 作为输入参数。

## 单线程

```
**import** **requests
from** **datetime** **import** **datetime**input_ids = list(range(1, 84)) * 5
# Increase the items of the list to simulate higher load
response_list = ['-'] * **len**(input_ids)**def** **request_person**(person_id, idx):
    url = f'[https://swapi.dev/api/people/{person_id}'](https://swapi.dev/api/people/{person_id}')
    **try**:
        r = requests.get(url)
        response_list[idx] = r.json()['name']
    **except** ***Exception*** **as** e:
        returnstart = datetime.now()
**for** idx, _id **in** enumerate(input_ids):
    **request_person**(_id, idx)print(datetime.now() - start)***>> 0:01:04.925463***
```

现在让我们用 ThreadPoolExecutor 将它转换成一个多线程任务:

## 多线程

```
**import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime**input_ids = list(range(1, 84)) * 5
response_list = ['-'] * len(input_ids)**def** **request_person**(person_id, idx):
    url = f'[https://swapi.dev/api/people/{person_id}'](https://swapi.dev/api/people/{person_id}')
    **try**:
        r = requests.get(url)
        response_list[idx] = r.json()['name']
    **except** ***Exception*** **as** e:
        returnstart = datetime.now()
**with** ThreadPoolExecutor(max_workers=20) as executor:    
    [executor.submit(**request_person**, person_id=_id, idx=idx) for idx, _id **in** enumerate(input_ids)]**print**(f'Time taken: {datetime.now() - start}')***>> 0:00:10.516984***
```

所以基本上，只增加几行额外的代码，我们就可以减少 7%的程序执行时间。

# 结论

有许多方法可以将并发性引入我们的 Python 程序，选择最佳方法将取决于您的具体用例。希望本文中的例子有助于您了解这个主题，并帮助您在以后的案例中进行复制。

> 感谢阅读这篇文章！有问题就在下面留言评论吧！