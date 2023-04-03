# 使用 Python 从抖音下载视频

> 原文：<https://medium.com/analytics-vidhya/download-tiktoks-with-python-dcbd79a5237f?source=collection_archive---------1----------------------->

![](img/0fdde0c5543c9946ee7ddef5cbbc1389.png)

照片由[阿诺德·弗朗西斯卡](https://unsplash.com/@clark_fransa)在 [Unsplash](https://unsplash.com/photos/f77Bh3inUpE) 拍摄

在过去的几个月里，抖音的受欢迎程度迅速上升。你可以用抖音的数据做很多项目，在这个项目中，我们将从抖音下载视频。

如果你更喜欢视觉学习，看看我为使用这个 API 制作的视频。我将在那个频道发布更多与逆向工程网站相关的内容，如果你对此感兴趣，请订阅。此外，请随意查看我的另一个 [YouTube 频道](https://www.youtube.com/c/davidteather)，这是与编程和工程相关的更有趣的内容:)

为了能够从 python 中检索抖音数据，运行以下命令来安装一个库来帮助我们。

```
pip install TikTokApi
```

然后，您应该运行以下命令来安装剧作家

```
python -m playwright install
```

我们需要找到 verifyFp 值。要做到这一点，去 tiktok.com 在你的浏览器。

点击 url 旁边的挂锁，如下图所示

![](img/dce2df33853f7e316fdf2bd230660121.png)

然后点击 cookies 显示你当前的 cookies

![](img/792c9be8bb2a2c1ce5d147da0c57271e.png)

转到[www.tiktok.com](http://www.tiktok.com)->cookie->s _ v _ webid，然后复制 s_v_webid cookie 的内容并粘贴到 verifyFp 中

![](img/9d74e543e2c8c32e7db92b8eef7299c9.png)

让我们创建一个新的 python 文件，您可以随意命名它。我们将进行测试，以确保我们能够从 python 中检索抖音数据。在 python 脚本中输入以下内容。

```
from TikTokApi import TikTokApi
import string
import random
did=''.join(random.choice(string.digits) for num in range(19))
verifyFp="verify_YOUR_VERIFYFP_HERE"
api = TikTokApi.get_instance(custom_verifyFp=verifyFp, custom_did=did)
print(api.trending())
```

运行您的 python 脚本，希望它将以 python 字典格式输出数据。如果脚本没有做到这一点，请访问 TikTokApi 的 GitHub 页面,您的问题很可能已经在已关闭的问题部分得到解决，只需搜索您可能遇到的任何错误。

> verifyFp 令牌似乎在大约 2 小时后过期，如果你在一个 verifyFp 下做了太多，它可能会抑制你的请求。目前，你可以通过做`*api = TikTokApi.get_instance(custom_verifyFp=verifyFp, use_test_endpoints=True*`来规避这个问题，但是我不能保证这在将来会有效

我们现在将修改之前的 python 脚本，以便它可以下载抖音趋势分析。

```
from TikTokApi import TikTokApi
import string
import random
did=''.join(random.choice(string.digits) for num in range(19))
verifyFp="verify_YOUR_VERIFYFP_HERE"
api = TikTokApi.get_instance(custom_verifyFp=verifyFp, custom_did=did)
print(api.trending())
```

现在我们必须实际保存来自抖音的视频。首先，我们将使用 python 创建一个名为 downloads 的文件夹，如果它还不存在的话。

```
from pathlib import PathPath("downloads").mkdir(exist_ok=True) # creates folder
```

我们现在将遍历 tiktoks 变量中的数据，并将所有视频下载到 downloads 文件夹中。

```
for i in range(len(tiktoks)):
    data = api.get_Video_By_TikTok(tiktoks[i])# bytes of the video
    with open("downloads/{}.mp4".format(str(i)), 'wb') as output:
        output.write(data) # saves data to the mp4 file
```

你完了！如果你运行这个脚本，它应该下载 30 本 TikToks 到下载目录。

如果您想增加每次运行脚本时下载的 TikToks 的数量，请更改下面一行

```
tiktoks = api.trending()
```

到下面

```
count = 100
tiktoks = api.trending(count=count)
```

现在你可以把 100 换成你想退多少。

> 注意:最大数量约为 2，000

可以通过代理服务器发出请求，这样抖音就无法阻止你的 IP 地址发出请求。

> 注意:许多免费代理不起作用，因为抖音已经阻止这些 IP 地址发出太多请求。

为了允许我们的脚本通过代理发出请求，让我们在脚本的顶部用我们的代理 IP 地址创建一个新变量。

```
proxy_address = "http://xxx.xxx.xxx:XXXX"
```

或者 API 支持以下用户身份验证

```
proxy_address = "http://user:password@xxx.xxx.xxx:XXXX"
```

> 注意:确保您的代理使用正确的 http 方案(https 或 http)。

要修改我们的脚本，只需将`proxy=proxy_address`作为参数添加到下面几行中。

```
tiktoks = api.trending(count=count)data = api.get_Video_By_TikTok(tiktoks[i])
```

这样它们看起来就像

```
tiktoks = api.trending(count=count, proxy=proxy_address)data = api.get_Video_By_TikTok(tiktoks[i], proxy=proxy_address)
```

> 注意:通过代理发出的请求比没有代理时要慢，并且对于大的计数值会花费相当多的时间。

没有代理的完整脚本

```
from TikTokApi import TikTokApi
import string
import random
did=''.join(random.choice(string.digits) for num in range(19))
verifyFp="verify_YOUR_VERIFYFP_HERE"
api = TikTokApi.get_instance(custom_verifyFp=verifyFp, custom_did=did)
print(api.trending())count = 100tiktoks = api.trending(count=count)from pathlib import PathPath("downloads").mkdir(exist_ok=True)for i in range(len(tiktoks)): data = api.get_Video_By_TikTok(tiktoks[i])# bytes of the video with open("downloads/{}.mp4".format(str(i)), 'wb') as output: output.write(data)
```

代理的完整脚本

```
from TikTokApi import TikTokApi
import string
import random
did=''.join(random.choice(string.digits) for num in range(19))
verifyFp="verify_YOUR_VERIFYFP_HERE"
api = TikTokApi.get_instance(custom_verifyFp=verifyFp, custom_did=did)
print(api.trending())# Change the line belowproxy_address = "http://xxx.xxx.xxx:XXXX"count = 100tiktoks = api.trending(count=count, proxy_address)from pathlib import PathPath("downloads").mkdir(exist_ok=True)for i in range(len(tiktoks)):data = api.get_Video_By_TikTok(tiktoks[i], proxy_address)with open("downloads/{}.mp4".format(str(i)), 'wb') as output:output.write(data)
```

感谢你阅读我的文章，希望对你有所帮助！