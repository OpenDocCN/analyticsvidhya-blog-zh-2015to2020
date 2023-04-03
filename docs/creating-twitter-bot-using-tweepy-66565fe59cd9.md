# 使用 Tweepy 创建 Twitter Bot

> 原文：<https://medium.com/analytics-vidhya/creating-twitter-bot-using-tweepy-66565fe59cd9?source=collection_archive---------20----------------------->

自动化不那么无聊的东西

![](img/8cfbaca69643a7e2ee194931eb605e6d.png)

Twitter 图标

自动化我们日常无聊的任务是一回事，自动化你喜欢做的事情是完全不同的另一回事。同样的任务，我试图完成一项我喜欢的任务，即活跃在社交媒体上。是的，我试图通过使用一个名为 [Tweepy 的流行 python 库来自动化我在 twitter 上的活动。](https://www.tweepy.org/)

[Tweepy](https://www.tweepy.org/) 是一个非常有趣的 python 库，它使得与 Python 开发者 API 的交互变得非常容易，并且隐藏了许多低级细节，这些细节对于一个刚刚开始使用 REST API 的初学者来说可能是难以理解的。

首先，我们需要用 [twitter](https://developer.twitter.com/en) 创建一个开发者账户，然后你可以创建一个应用程序，提供我们在创建机器人时必须使用的以下凭证。

*   访问令牌
*   访问令牌密码
*   API_KEY
*   API_SECRET_KEY

这些凭证应该保密，不应该以任何方式共享，将这些细节放在源代码中并推送到 GitHub 是一个常见的错误，因此这些应该只在环境变量中使用。

```
__API_KEY = os.environ['API_KEY']
__API_SECRET_KEY = os.environ['API_SECRET_KEY']
__ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
__ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']
```

还有一件事要开始，我们需要通过 pip 安装 [tweepy](https://www.tweepy.org/) 库

要快速完成这项工作，你可以使用我从 Github 获得的示例机器人，这里是[链接](https://github.com/amanchourasiya/twitter-bot)。

```
git clone [https://github.com/amanchourasiya/twitter-bot](https://github.com/amanchourasiya/twitter-bot)
export ACCESS_TOKEN="<access token from twitter developer account>"
export ACCESS_TOKEN_SECRET="<access token from twitter developer account>"
export API_KEY="<api key from twitter developer account>"
export API_SECRET_KEY="<api secret key from twitter developer account>"
python3 bot.py
```

这个机器人目前被编程为喜欢和转发来自特定帐户的推文，并且这个机器人还根据特定的过滤器实时过滤推文。

# 此机器人中使用的组件

![](img/dcf95ff66d4db520d47ed24e5772f101.png)

这个机器人没有很多组件，但它具有足够的可扩展性，可以针对不同的目的进行定制。

# 1.正在为 Twitter API 创建授权配置

使用 Tweepy 创建到 twitter API 的认证连接需要我们从 twitter 开发者帐户获得的所有四个秘密值。

```
class Bot:
    def __init__(self):
        try:
            __API_KEY = os.environ['API_KEY']
            __API_SECRET_KEY = os.environ['API_SECRET_KEY']
            __ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
            __ACCESS_TOKEN_SECRET = os.environ['ACCESS_TOKEN_SECRET']auth = tweepy.OAuthHandler(__API_KEY, __API_SECRET_KEY)
            auth.set_access_token(__ACCESS_TOKEN, __ACCESS_TOKEN_SECRET)
            self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
            self.api.verify_credentials()
            print('Authentication successful')
        except KeyError:
            print('Twitter credentials not supplied.')
        except Exception as e:
            print(e)
```

它处理创建与 twitter API 的 OAuth 连接，并设置警报和通知的速率限制，twitter 有一些设置的 API 使用限制，如果我们超过该限制，那么我们的 API 处理程序将等待该时间，然后再次开始调用 twitter API。

# 2.创建效用函数。

应该为这个机器人的基本使用创建一些实用函数，这些可能包括

*   获得前 20 条推文。
*   获得 10 个热门标签。
*   从特定用户获取最近的推文。
*   获取所有关注者的列表。

这个列表可以是无穷无尽的，因为这些是 twitter API 的无限用例，使用这些实用函数，我们可以调整我们的机器人，以基于这些函数的结果触发一些函数。

例如，假设我们计划制作一个功能，如果用户转发推文，我们会自动关注该用户，这样我们可以制作多个功能并定期运行它们。

# 3.从推特上传实时数据。

我们可以使用 Teeepy 的 StreamListerner 对象从 twitter 新闻提要中流式传输和过滤实时数据，并在此基础上指示我们的机器人做出一些决定。这些数据也可以用来对 Twitter 数据进行实时分析。

```
class WorkerStreamListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        self.me = api.me()def on_status(self, tweet):# Check if this tweet is just a mention
        print(f'{tweet.user.name}: {tweet.text}')print(f'{tweet.user.id}')# Check if tweet is reply
        if tweet.in_reply_to_status_id is not None:
            return# Like and retweet if not done already
        if not tweet.favorited:
            try:
                tweet.favorite()
            except Exception as e:
                print(f'Exception during favourite {e}')
        if not tweet.retweeted:
            try:
                tweet.retweet()
            except Exception as e:
                print(f'Exception during retweet {e}')print(f'{tweet.user.name}:{tweet.text}')def on_error(self, status):
        print('Error detected')stream.filter(track=["Hacking"], languages=["en"], is_async=True)
```

就像这样，我们可以创建流对象并根据一些目标关键字过滤数据，这个流监听器正在做的是，每当有任何包含关键字 **Hacking** 的推文时，它就会自动喜欢并自动转发该推文。

这只是 twitter 机器人功能的一个示例，通过研究这个 API 的所有特性可以实现更多功能。当我写这篇博客时，我的 twitter-bot 项目也在第一阶段。

*最初发表于*[T5【https://www.amanchourasiya.com】](https://www.amanchourasiya.com/blog/creating-twitter-bot-using-tweepy/)*。*

关注我的更多博客[https://twitter.com/TechieAman](https://twitter.com/TechieAman)

个人博客 https://www.amanchourasiya.com