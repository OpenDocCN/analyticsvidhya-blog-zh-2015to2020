# 用 BeautifulSoup 抓取 Youtube 视频(python)

> 原文：<https://medium.com/analytics-vidhya/scraping-youtube-video-with-beautifulsoup-python-d30598285965?source=collection_archive---------2----------------------->

![](img/a0c3ab1fc19febcd4e9ca976a38a1741.png)

最近我沉迷于 youtube 视频的洞察力，因此我试图用我最喜欢的 python- BeautifulSoup 包来抓取这个网站。现有的爬虫并没有给我一个明确的方向，最重要的是我发现它们都没有涵盖如何提取视频类别。所以我决定从头开始做自己的爬虫。

嗯，我不擅长文字，所以没有延误，我会去我的编码。

首先，让我们从加载所需的存储库开始:

```
from bs4 import BeautifulSoup #for scraping
import requests               #required for reading the file
import pandas as pd           #(optional) Pandas for dataframes 
import json                   #(optional) If you want to export json
import os
```

既然已经成功加载了所需的存储库，那么让我们开始我们的主要目标，搜索所需的信息。对于那些不熟悉 BeautifulSoup 的人，我会建议他们去阅读软件包的文档。有教程定义了如何使用 BeautifulSoup 进行刮擦。

由于这只是一个爬虫，我使用了 youtube 链接的用户输入。但是我在 youtube 的链接列表中使用了相同的代码，这很好，你可以根据自己的需要修改代码。

```
url = input('Enter Youtube Video Url- ') # user input for the link
Vid={}
Link = url
source= requests.get(url).text
soup=BeautifulSoup(source,'lxml')
div_s = soup.findAll('div')Title = div_s[1].find('span',class_='watch-title').text.strip()
Vid['Title']=Title
Vid['Link']=Link
Channel_name = div_s[1].find('a',class_="yt-uix-sessionlink spf-link").text.strip()
Channel_link = ('[www.youtube.com'+div_s[1].find('a',class_=](http://www.youtube.com'+div_s[1].find('a',class_=)"yt-uix-sessionlink spf-link").get('href'))
Subscribers = div_s[1].find('span',class_="yt-subscription-button-subscriber-count-branded-horizontal yt-subscriber-count").text.strip()
if len(Channel_name) ==0
    Channel_name ='None'
    Channel_link = 'None'
    Subscribers = 'None'
Vid['Channel']=Channel_name
Vid['Channel_link']=Channel_link
Vid['Channel_subscribers']=Subscribers
```

对于类别，我发现这种提取类别信息的方法不起作用，这主要是因为类别信息会随着语言设置的变化而变化。所以我找到的最好的方法是采用类别 Id，它是数字，并且只能是这 15 个类别中的一个(带有相应的类别 Id:类别名称):

1:'电影&动画'，
2:'汽车&车辆'，
10:'音乐'，
15:'宠物&动物'，
17:'体育'，
19:'旅游&事件'，
20:'博彩'，
22:'人&博客'，
23:'喜剧'，
24:'娱乐'，
25:'新闻&政治'，【T10

一旦定义了这一点，剩下的任务就简单了:

```
Category_index = {
     1 : 'Film & Animation',
     2 : 'Autos & Vehicles',
     10 : 'Music',
     15 : 'Pets & Animals',
     17 : 'Sports',
     19 : 'Travel & Events',
     20 : 'Gaming',
     22 : 'People & Blogs',
     23 : 'Comedy',
     24 : 'Entertainment',
     25 : 'News & Politics',
     26 : 'Howto & Style',
     27 : 'Education',
     28 : 'Science & Technology',
     29 : 'Nonprofits & Activism'}
Sp = div_s[1].text.split(':')
subs = 'categoryId'
for j in range(len(Sp)):
    if subs in Sp[j]:
        value =int(Sp[j+1].split(',')[0])
Video_category=Category_index[value]        
Vid['Category']=Video_categoryView_count = div_s[1].find(class_= 'watch-view-count')
View_count = View_count.text.strip().split()[0]
Vid['Views']=View_countLikes = div_s[1].find('button',class_="yt-uix-button yt-uix-button-size-default yt-uix-button-opacity yt-uix-button-has-icon no-icon-markup like-button-renderer-like-button like-button-renderer-like-button-unclicked yt-uix-clickcard-target yt-uix-tooltip" ).text.strip()
Vid['Likes']=Likes
Dislikes = div_s[1].find('button',class_="yt-uix-button yt-uix-button-size-default yt-uix-button-opacity yt-uix-button-has-icon no-icon-markup like-button-renderer-dislike-button like-button-renderer-dislike-button-unclicked yt-uix-clickcard-target yt-uix-tooltip" ).text.strip()
Vid['Dislikes']=DislikesRelated_videos = div_s[1].findAll('a',class_='content-link spf-link yt-uix-sessionlink spf-link')
Title_Related=[]
Link_Related =[]
for i in range(len(Related_videos)):
    Title_Related.append(Related_videos[i].get('title'))
    Link_Related.append(Related_videos[i].get('href'))
Related_dictionary = dict(zip(Title_Related, Link_Related))    
Vid['Related_vids']=Related_dictionary
```

成功抓取所需信息后，存储在 Vid 字典中，您可以将其导出为 JSON 文件:

```
with open('vfile.json', 'w', encoding='utf8') as ou:
    json.dump(Vid, ou, ensure_ascii=False)
```

或数据帧:

```
df = pd.DataFrame(Vid, index =[0])
```

vfile.json 将保存在脚本的同一个目录中，或者如果您想继续进行一些探索性的数据分析，您可以继续使用 data frames——我个人认为这更有用。

注意:这里描述的所有代码都可以在我的 github 账户中找到——[https://github.com/DiproMondal/Youtube_crawler](https://github.com/DiproMondal/Youtube_crawler)

如果任何人有任何问题/意见，随时联系我这里或邮件:dipro.mondal12ms@gmail.com

谢谢你