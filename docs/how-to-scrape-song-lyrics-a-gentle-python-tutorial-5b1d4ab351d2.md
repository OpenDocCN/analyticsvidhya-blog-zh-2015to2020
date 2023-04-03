# 如何刮歌词:温柔教程

> 原文：<https://medium.com/analytics-vidhya/how-to-scrape-song-lyrics-a-gentle-python-tutorial-5b1d4ab351d2?source=collection_archive---------2----------------------->

使用 Genius.com API+Beautiful Soup 获取歌词并写入本地文本文件。关于您可以用这些数据做的有趣事情的未来帖子。

这将是一个非常简单的教程，你可以按照它将歌词收集到一个本地`.txt`文件中

本教程唯一困难的是在没有语法高亮的介质上阅读代码

# 先决条件:

*   获得 Genius.com API 密钥(免费！):在这里按照[的指示](https://docs.genius.com/#/getting-started-h1)
*   我用 Python3

```
GENIUS_API_TOKEN='YOUR-TOKEN-HERE'
```

# 进口

```
# Make HTTP requests
import requests# Scrape data from an HTML document
from bs4 import BeautifulSoup# I/O
import os# Search and manipulate strings
import re
```

# 1.获取某个艺术家指定数量歌曲的 Genius.com URL 列表

```
# Get artist object from Genius API
def request_artist_info(artist_name, page):
    base_url = '[https://api.genius.com'](https://api.genius.com')
    headers = {'Authorization': 'Bearer ' + GENIUS_API_TOKEN}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': artist_name}
    response = requests.get(search_url, data=data, headers=headers)
    return response# Get Genius.com song url's from artist object
def request_song_url(artist_name, song_cap):
    page = 1
    songs = []

    while True:
        response = request_artist_info(artist_name, page)
        json = response.json() # Collect up to song_cap song objects from artist
        song_info = []
        for hit in json['response']['hits']:
            if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
                song_info.append(hit)

        # Collect song URL's from song objects
        for song in song_info:
            if (len(songs) < song_cap):
                url = song['result']['url']
                songs.append(url)

        if (len(songs) == song_cap):
            break
        else:
            page += 1

    print('Found {} songs by {}'.format(len(songs), artist_name))
    return songs

# DEMO
request_song_url('Lana Del Rey', 2)
```

例如，这可能会返回:

```
['https://genius.com/Lana-del-rey-young-and-beautiful-lyrics',
 'https://genius.com/Lana-del-rey-love-lyrics']
```

![](img/d19b1142b2295a9dd5b79b9e7702cfba.png)

文章灵感来自于听 LDR 的新专辑

# 2.从 URL 的获取歌词

```
# Scrape lyrics from a Genius.com song URL
def scrape_song_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')
    lyrics = html.find('div', class_='lyrics').get_text()
    #remove identifiers like chorus, verse, etc
    lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
    #remove empty lines
    lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])         
    return lyrics# DEMO
print(scrape_song_lyrics('[https://genius.com/Lana-del-rey-young-and-beautiful-lyrics'](https://genius.com/Lana-del-rey-young-and-beautiful-lyrics')))
```

这将返回:

```
I've seen the world, done it all
Had my cake now
Diamonds, brilliant, and Bel Air now
Hot summer nights, mid-July
When you and I were forever wild
The crazy days, city lights
The way you'd play with me like a child
Will you still love me when I'm no longer young and beautiful?
Will you still love me when I got nothing but my aching soul?
I know you will, I know you will, I know that you will
Will you still love me when I'm no longer beautiful?
I've seen the world, lit it up as my stage now
Channeling angels in the new age now
Hot summer days, rock and roll
The way you'd play for me at your show
And all the ways I got to know
...[etc.]
```

# 3.循环浏览所有 URL，并将歌词写到一个文件中

```
def write_lyrics_to_file(artist_name, song_count):
    f = open('lyrics/' + artist_name.lower() + '.txt', 'wb')
    urls = request_song_url(artist_name, song_count)
    for url in urls:
        lyrics = scrape_song_lyrics(url)
        f.write(lyrics.encode("utf8"))
    f.close()
    num_lines = sum(1 for line in open('lyrics/' + artist_name.lower() + '.txt', 'rb'))
    print('Wrote {} lines to file from {} songs'.format(num_lines, song_count))

# DEMO  
write_lyrics_to_file('Kendrick Lamar', 100)
```

这会将一百首 Kendrick Lamar 歌曲的歌词写入一个文件`./lyrics/kendrick lamar.txt`

```
Found 100 songs by Kendrick Lamar
Wrote 8356 lines to file from 100 songs
```

这里有一个年轻暴徒的输出示例:【https://pastebin.com/raw/TKqrGiTg

# 后续步骤

TBD 我打算用这个轻量级工具做什么新颖的应用，但这里有一些启发我的想法:

*   [在 Kanye West 的整个唱片集上训练的说唱歌曲写作递归神经网络](https://github.com/robbiebarrat/rapping-neural-network)
*   [同样的事情，但泰勒·斯威夫特](https://www.google.com/search?q=song+lyric+generatio+python&oq=song+lyric+generatio+python&aqs=chrome..69i57.4779j0j4&sourceid=chrome&ie=UTF-8)
*   [又一篇好的抒情代文章](https://towardsdatascience.com/text-predictor-generating-rap-lyrics-with-recurrent-neural-networks-lstms-c3a1acbbda79)
*   [情感分析](https://www.kdnuggets.com/2018/09/sentiment-analysis-adele-songs.html)
*   基于来自两个或更多艺术家的数据训练的歌词生成(Kanye + TSwift，Young Thug +拉娜·德尔·雷，Travis Scott+Beach Boys…go wild my children)
*   [歌词生成+音频生成= AI 生成的歌曲？](https://towardsdatascience.com/neural-networks-for-music-generation-97c983b50204)