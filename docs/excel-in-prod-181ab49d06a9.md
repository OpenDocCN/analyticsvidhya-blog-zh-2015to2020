# 擅长生产

> 原文：<https://medium.com/analytics-vidhya/excel-in-prod-181ab49d06a9?source=collection_archive---------15----------------------->

## 或者*如何使用水管工*或*微 excel 微服务*将 excel 整合到生产 API 中

![](img/08df472421b3555b40b2eab4fe9aba9e.png)

照片由[米卡·鲍梅斯特](https://unsplash.com/@mbaumi?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

*最初发表于*[*http://josiahparry.com*](http://josiahparry.com/post/2020-03-01-excel-in-prod/)*。*

我最近有一次谈话，谈到了使用自动解析 Excel 文档来管理数据科学资产。这引出了一些非常有趣的观点:

1.  Excel 有时是不可避免的，我们需要接受这一点。
2.  如何才能将 Excel 融入到生产中？

> *注意，现在不是时候💩在 Excel 上。它服务于非常真实的商业目的，不幸的是不是每个人都能学会编程😕。对于 h8rs 来说，这里有一个有趣的问题:几乎每个总统选举活动的数据程序都是基于 Google Sheets 的背面。*

在这篇文章中，我开始探索是否以及如何将 Excel 整合到生产代码中。这里使用的代码请参见 [GitHub 库](https://github.com/JosiahParry/plumber-excel)。

生产化代码是什么意思——也就是把它放到产品中？这是什么意思，没有一个统一定义，每个组织都有不同的操作方法。

> *可操作化的定义*，至少从社会科学的角度来看，是如何定义一个事物，以便对该事物有一个共同的理解。**

*格雷格·威尔逊将其定义为“不会让运营团队哭泣的代码”(强调他的)。这是我最喜欢的定义，因为它是异想天开的、讽刺的、诚实的，并且承认代码将不得不离开数据科学的核心圈子。*

*据我所知，当前的数据科学讨论强调使用 RESTful APIs 作为最好的，或者至少是主导的，生产代码的方式。*

*API 是一个应用程序编程接口。当我第一次学习什么是 API 时，我叔叔告诉我把它们想象成“与其他机器对话的机器”那并不遥远！*

*RESTful API 是一种特殊的 API。坦率地说，我不知道这到底意味着什么。据我理解，REST 实际上是一种自以为是的 API 架构方式。RESTful APIs 使用 HTTP 请求，这使得它们非常容易访问。RESTful APIs 是开发微服务的关键，微服务是将代码放入产品的核心。在 python 生态系统中，Flask 是制作微服务的领先库之一。在 R 空间中，一个名为`plumber`的包承担了这个角色。*

*我们可以设想一个假设的场景，其中我们通过一些数据收集过程接收 Excel 文件。一旦收到该 Excel 文件，它将用于启动一些其他流程，例如报告编写或数据处理。人们通常会创建闪亮的应用程序来提供上传和数据输入的 UI。当我们想开发一个以用户为中心的界面时，这真的很棒。但是当我们想要自动化这个过程或者至少让其他工具可以使用这个过程的时候呢？这时，我们可以求助于 plumber 来创建一个微服务来处理这个问题。*

*上图(用制作)展示了两种不同的方法。首先，我们将收到 Excel 文件。从那里，我们可能希望将文件上传到共享驱动器、数据库或两者。或者，我们可能不想存储数据，而是想立即使用它。*

*从 API 开发的角度来看，我们可以将每个流程想象成一个 API *端点*。端点本质上是一个 url，它表明每个应用程序交互将在哪里发生。在这个小例子中，我们将创建两个端点:`/read_excel`和`/upload`。第一个会，你猜对了，读取一个随请求发送的 Excel 文件。第二个将上传所述文件。*

*在我们着手创建 API 之前，我们需要首先弄清楚如何通过 API 发送文件。在我们弄清楚这一点之前，我们需要知道我们可以向 API 发出什么类型的请求。因为 REST API 将是一个 HTTP API，所以我们必须知道使用 HTTP 协议可以发出什么类型的请求。有 7 种 HTTP 请求类型。*

*坦白地说，我不记得`HEAD`、`PATCH`和`OPTIONS`是做什么的——如果你不使用它，你就会失去它，对吗？对于超级简单的 API，我们只需要知道`GET`和`POST`请求——不要让我做任何事情，我不可信。*

*`GET`用于*从资源中获取*数据。您可以将键值对作为参数传递给`GET`请求。"`GET`请求仅用于请求数据(不能修改)."你永远不应该通过`GET`请求发送敏感信息。*

*这就把我们带到了`POST`方法。`POST`方法用于向服务器发送数据，以便创建、修改或更新资源。当你有很多参数要发送，或者如果它们是敏感的，或者如果你需要通过 API 发送一个文件，使用`POST`。*

*在研究这个 API 设计时，我有三个问题。*

1.  *你是如何通过 HTTP 请求发送文件的？*
2.  *一旦我们发送了它，我们如何访问文件，它去了哪里？*
3.  *我们如何将 API 中的数据传递给 R？*

*我不会说 Linux，所以祝福`httr`让这变得简单(有点)。`httr`包含发布 excel 文件的两个核心功能。有`POST()`用于发出 post 请求，还有`upload_file()`用于上传 POST 请求中的文件。*

> *我们能不能花点时间来欣赏一下函数有时是多么完美的命名？越不言自明越好。*

*如果你没有太多使用`httr`制作请求的经验，我建议你从[快速入门简介](https://cran.r-project.org/web/packages/httr/vignettes/quickstart.html)开始。*

*我们的 POST 请求的结构如下所示*

```
*POST(end_point, body = list(param = upload_file("file-path.ext")) )*
```

# *构建第一个端点*

*现在我们知道*文件将如何发送。但是困难的部分实际上是构建将接收它的管道工 API 端点。关于如何通过水管工上传文件有相当多的讨论。还好， [@krlmlr](https://github.com/krlmlr) 指出了`mime::parse_multipart()`可以用来处理请求中发送的文件。**

> *注意:MIME 是通过互联网发送文件的标准方式。我对哑剧类型一无所知，非常欣赏谢一辉、杰弗里·霍纳和边·贝勒用这个包所做的工作，他们把所有这些都抽象了出来。*

*`parse_multipart()`将接受传入的请求并返回一个命名列表。对我们来说最重要的是，返回的对象包含了我们发布到临时位置的文件。在结果列表中是临时文件的路径。在我们的管道工函数定义中，我们解析请求，并取出`datapath`。保存的路径然后被提供给`readxl::read_excel()`,后者返回一个 tibble！*

```
*#* Read excel file #* @post /read_excel function(req) { multipart <- mime::parse_multipart(req) 
  fp <- purrr::pluck(multipart, 1, "datapath", 1)
  readxl::read_excel(fp) }pr <- plumber::plumb("plumber.R") 
pr$run(host = "127.0.0.1", port = 5846)# start a background job using the RStudio job launcher rstudioapi::jobRunScript(
  path = file.path(here::here(), "activate.R"),
  workingDir = here::here()
)*
```

*![](img/f81427e6793e5a86796be0111f805ed9.png)*

```
*#* Read excel file 
#* @post /read_excel 
function(req) { 
  mime::parse_multipart(req) 
}*
```

# *创建 API 包装*

*创建 API 包装器是我最喜欢的活动之一，因为它相当简单，而且感觉超级强大💪🏼。如前所述，我们创建 POST 请求所需要做的就是指定在哪里发出请求(端点)，并为其提供一些参数。*

*在后台启动 API。*

```
*# start a background job using the RStudio job launcher rstudioapi::jobRunScript(
  path = file.path(here::here(), "activate.R"),
   workingDir = here::here())*
```

*我们首先用端点定义一个名为`b_url`(基本 url)的对象。接下来，我们在`upload_file()`命令中指定想要上传的文件的路径。在存储库中，我已经包含了`test.xls`，它包含了美国社区调查(社会科学，amirite？).请注意，上传的文件是`body`参数中命名列表的一部分。任何需要传递给 API 的参数都需要在提供给`body`的列表中定义。我*认为*上传文件的名称需要与 plumber API ( `req`)中定义的名称相匹配。我可能错了，但为了安全起见！*

```
*library(httr) 
library(tidyverse)

# define the url b_url <- "http://127.0.0.1:5846/read_excel" # make the request! 
req <- POST(b_url, body = list(req = upload_file("data/test.xls")))*
```

*我们现在已经上传了文件并提出了我们的请求！虽然如果我们不能访问数据，这个请求对我们没有用😮。我们可以使用`httr::content()`获得请求的内容。我设置了`type = "text/json"`,因为我发现将 json 转换成 tibble 比转换成命名列表更容易。*

```
*# get json 
res_json <- content(req, type = "text/json")

# show the first 50 characters of resultant json 
strtrim(res_json, 50)## [1] "[{\"statefip\":1,\"costelec\":7440,\"costgas\":5760,\"cos"*
```

*要将这个 json 放入 tibble，将使用`jsonlite::fromJSON()`和`tibble::as_tibble()`。*

```
*res <- content(req, type = "text/json") %>% 
  jsonlite::fromJSON() %>% 
  as_tibble() glimpse(res)## Observations: 52 ## Variables: 27
## $ statefip <int> 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18... 
## $ costelec <int> 7440, 7080, 7320, 6600, 7800, 6240, 7800, 6720, 648... 
## $ costgas <int> 5760, 6240, 4800, 5880, 5400, 5520, 6840, 6360, 528... 
## $ costwatr <int> 3100, 3400, 3800, 3100, 4100, 3300, 3100, 2800, 360... 
## $ costfuel <int> 3900, 7100, 2800, 3300, 2900, 2400, 5700, 3500, 430... 
## $ condofee <int> 1100, 690, 1000, 870, 1200, 920, 940, 950, 1400, 14... 
## $ rent <int> 2400, 2800, 2900, 2000, 3900, 3200, 3300, 2800, 390... 
## $ proptx99 <int> 66, 69, 69, 67, 69, 69, 69, 69, 69, 69, 69, 69, 69,... 
## $ propinsr <int> 5400, 7500, 5000, 5200, 8100, 7200, 9800, 6000, 770... 
## $ mortamt1 <int> 3400, 4000, 5000, 3600, 7300, 5400, 7400, 4000, 700... 
## $ mortamt2 <int> 1600, 2900, 2400, 2200, 3900, 3600, 2900, 2700, 540... 
## $ moblhome <int> 4300, 7900, 10300, 4800, 13900, 12000, 8000, 10100,... 
## $ rooms <int> 16, 16, 15, 15, 14, 17, 17, 16, 16, 14, 18, 14, 17,... 
## $ bedrooms <int> 8, 8, 6, 6, 7, 7, 7, 7, 7, 6, 7, 7, 7, 7, 8, 8, 7, ... 
## $ valueh <int> 1551000, 1945000, 2387000, 1901000, 6288000, 348100... ## $ incwage <int> 391000, 439000, 412000, 422000, 565000, 498000, 718... ## $ incbus00 <int> 335000, 256000, 280000, 295000, 372000, 320000, 448... ## $ incinvst <int> 299000, 96000, 306000, 263000, 353000, 290000, 3230... ## $ incretir <int> 109000, 148000, 138000, 113000, 169000, 129000, 158... ## $ incother <int> 64000, 33200, 73000, 67000, 78000, 76000, 79000, 72... ## $ incwelfr <int> 10200, 11800, 12700, 11300, 18900, 14500, 19000, 75... ## $ incsupp <int> 22700, 24000, 25800, 22700, 25400, 26200, 26600, 24... ## $ incss <int> 34500, 32400, 34400, 35100, 35300, 35100, 37700, 36... ## $ age <int> 93, 91, 93, 94, 94, 93, 96, 92, 94, 95, 92, 96, 94,... ## $ trantime <int> 152, 162, 145, 129, 141, 157, 142, 157, 111, 150, 1... ## $ incbus00_min <int> -6900, -6500, -5000, -7800, -4800, -5800, -4800, -6... ## $ incincst_min <int> -1500, -800, -1600, -920, -2300, -1800, -1800, -140...*
```

*嘣！！！成功了。现在是时候把它变成一个函数了。为了使其通用化，我们需要让用户可以为`upload_file()`指定文件路径。*

```
*# define `post_excel()` 
post_excel <- function(file) { b_url <- "http://127.0.0.1:5846/read_excel"   req <- POST(b_url, body = list(req = upload_file(file)))    res <- content(req, type = "text/json") %>% 
             jsonlite::fromJSON() %>%
             tibble::as_tibble() res 
}*
```

*您已经为您的 API 创建了一个包装器！现在您已经有了一个正在运行的微服务，可以通过 R 包装器访问它。*

```
*post_excel("data/test.xls")## # A tibble: 52 x 27 
## statefip costelec costgas costwatr costfuel condofee rent proptx99 
## <int> <int> <int> <int> <int> <int> <int> <int> 
## 1 1 7440 5760 3100 3900 1100 2400 66 
## 2 2 7080 6240 3400 7100 690 2800 69
## 3 4 7320 4800 3800 2800 1000 2900 69 
## 4 5 6600 5880 3100 3300 870 2000 67 
## 5 6 7800 5400 4100 2900 1200 3900 69 
## 6 8 6240 5520 3300 2400 920 3200 69 
## 7 9 7800 6840 3100 5700 940 3300 69 
## 8 10 6720 6360 2800 3500 950 2800 69 
## 9 11 6480 5280 3600 4300 1400 3900 69 
## 10 12 6720 3480 3400 3600 1400 3300 69 
## # ... with 42 more rows, and 19 more variables: propinsr <int>, ## # mortamt1 <int>, mortamt2 <int>, moblhome <int>, rooms <int>, 
## # bedrooms <int>, valueh <int>, incwage <int>, incbus00 <int>, 
## # incinvst <int>, incretir <int>, incother <int>, incwelfr <int>, ## # incsupp <int>, incss <int>, age <int>, trantime <int>, 
## # incbus00_min <int>, incincst_min <int>*
```

*我们可以为`/upload`端点创建一个类似的函数。*

```
*upload <- function(file) {   b_url <- "http://127.0.0.1:5846/upload" usethis::ui_info(glue::glue("Copying {file} to /data/{file}"))  req <- POST(b_url, body = list(req = upload_file(file)))   invisible(req) 
}upload("data/test.xls")## ℹ Copying data/test.xls to /data/data/test.xls*
```

> **注:我建议使用{usethis}的* `*ui_*()*` *功能，为用户提供信息性消息。
> 第二个注意:如果你打算只允许一个文件上传一次，就像这个函数所做的那样，你可能实际上应该使用一个上传请求。**

*吧嗒吧嗒嘣。现在，您可以使用 plumber 创建一个能够处理 Microsoft Excel 文件的微服务。这不是一个小壮举！下一步是什么？您应该为新创建的 API 创建一个漂亮的小 python 包装器。python 包装器对您的团队来说将是一笔巨大的财富，现在您的基于 R 的工具可以被任何能够发出 HTTP 请求的人或事物访问！！！*