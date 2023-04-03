# 如何不用 Python 中的 API 生成城市的经纬度坐标？

> 原文：<https://medium.com/analytics-vidhya/how-to-generate-lat-and-long-coordinates-of-city-without-using-apis-25ebabcaf1d5?source=collection_archive---------1----------------------->

易于理解的代码

![](img/304d195fd718da5e9d12812916748d68.png)

[乔·塞拉斯](https://unsplash.com/@joaosilas?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

如果有人想用地理坐标(定义地球表面某点位置的纬度和经度)绘制地图。常见的坐标选择是[纬度](https://en.wikipedia.org/wiki/Latitude)，[经度](https://en.wikipedia.org/wiki/Longitude)。

![](img/f632da280aa26f5428e4781fcf51d357.png)

照片由[奥克萨娜 v](https://unsplash.com/@arttravelling?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

**纬度**线东西走向，相互平行。如果你往北走，纬度值会增加。最后，纬度值(Y 值)的范围在-90 度到+90 度之间

但是经度线是南北走向的。它们汇聚在两极。它的 X 坐标在-180 度到+180 度之间。

制图员用[度-分-秒(DMS)和十进制度](https://gisgeography.com/decimal-degrees-dd-minutes-seconds-dms/)来写球面坐标(纬度和经度)。对于度-分-秒，分的范围是从 0 到 60。例如，以度-分-秒表示的纽约市的地理坐标为:

*   纬度:北纬 40 度 42 分 51 秒
*   经度:西经 74 度 0 分 21 秒

我们也可以用十进制度数来表示地理坐标。这只是用不同格式表示相同位置的另一种方式。例如，这里是纽约市的十进制度数:

*   纬度:40.714 度
*   经度:-74.006 度

如果您想更深入地了解，请在此处阅读更多。

# 我们开始吧

我使用 Jupyter notebook 来运行本文中的脚本。

首先，我们将使用 PIP 安装像 nomist 和 geopy 这样的库。

```
pip install geopy 
pip install Nominatim
```

现在代码非常简单，我们只需要运行这个

案例 1:只提到城市名称

```
from geopy.geocoders import Nominatimaddress='Nagpur'
geolocator = Nominatim(user_agent="Your_Name")
location = geolocator.geocode(address)
print(location.address)
print((location.latitude, location.longitude))
```

运行上述代码后，这是我们将得到的输出。

```
Nagpur, Nagpur District, Maharashtra, 440001, India
(21.1498134, 79.0820556)
```

案例 2:同时提到国家和城市名称。

如果我们有国家名称和城市名称，我们也可以运行另一个代码。

```
from  geopy.geocoders import Nominatim
geolocator = Nominatim()city ="Agra"
country ="India"
loc = geolocator.geocode(city+','+ country)
print("latitude is :-" ,loc.latitude,"\nlongtitude is:-" ,loc.longitude)
```

输出是:

```
latitude is :- 27.1752554 
longtitude is:- 78.0098161
```

![](img/264572514f13b3d83bed92de0cffc752.png)

亚历克斯·佩雷兹在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

结论

我们也可以使用 googlemaps APIs 生成 lat 和 long，但是对于 API，您必须支付一些费用。

我希望这篇文章能帮助你并节省大量的时间。如果你有任何建议，请告诉我。

快乐编码。

来源:

【https://gisgeography.com/latitude-longitude-coordinates/ 

https://en.wikipedia.org/wiki/Geographic_coordinate_system/

![](img/32f9a2a401973e5f770c6ee9f638fbd8.png)

由 [Keegan Houser](https://unsplash.com/@khouser01?utm_source=medium&utm_medium=referral) 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片