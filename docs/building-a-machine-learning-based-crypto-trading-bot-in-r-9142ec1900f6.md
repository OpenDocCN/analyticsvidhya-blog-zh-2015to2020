# 用 R 语言构建一个基于机器学习的密码交易机器人

> 原文：<https://medium.com/analytics-vidhya/building-a-machine-learning-based-crypto-trading-bot-in-r-9142ec1900f6?source=collection_archive---------0----------------------->

![](img/5d4e5834be3f7d0657d33eedfaa602d7.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) 拍摄的照片

谁不想尝试构建一个可以击败市场的机器人呢？根据一些人的说法，市场应该是随机游走的，但这并没有阻止人们向市场投掷神经网络，因为为什么不呢？

构建一个机器人本身可能是一项有趣的任务，这最终促使我尝试一下。在这篇文章中，我将讨论任何机器学习任务中必不可少的第一步:获取数据。

现在我已经用北海巨妖交易密码，他们有一个功能良好的 API，可以用来收集他们所有的历史交易数据。在 R 中，它变成了与这个 API 交互并保存数据的任务。这个任务可以设置在云中，但是我决定在本地提取数据。下面的代码显示了如何从北海巨妖提取数据。我将从提取本身开始，因为它足够简单:

```
pair = "XETCZEUR"url <- paste0("[https://api.kraken.com/0/public/Trades?pair=](https://api.kraken.com/0/public/Trades?pair=)",pair,"&since=0")dat<-jsonlite::fromJSON(url)$result
dat[pair]
```

现在这只提取了 1000 行交易，所以为了继续提取，我们将进行另一个 API 调用:

```
url<-paste0('[https://api.kraken.com/0/public/Trades?pair=',pair,'&since=',dat$last](https://api.kraken.com/0/public/Trades?pair=',pair,'&since=',dat$last))

dat<-jsonlite::fromJSON(url)$result
```

正如你所看到的，这需要一段时间。北海巨妖 API 对你在一个时间段内可以调用多少次有规定，所以当循环时，我们必须增加一些等待时间。

为了提取数据而不必担心 API 调用失败和各种其他错误，我使用了下面的脚本。我可以推荐你自己把代码复制到 R 里。

```
library(httr)
library(jsonlite)
library(RCurl)
library(RSQLite)get_latest <- function(pairfolder,pair){
csvs<-list.files(paste0("D:/Data/Coins Data/",pairfolder,"/"))
if(is.na(csvs[1])){
  return(NA)
}
csvs_n <- as.numeric(gsub("[^[:digit:]]","",csvs))latest<-read.csv(paste0("D:/Data/Coins Data/",pairfolder,"/",pair,"_",max(csvs_n),".csv"))latest_time<-paste(as.numeric(latest[dim(latest)[1],4])* 1000000000)
return(latest_time)
}pair = "XETCZEUR"
pairfolder <- "ETC"csvs<-list.files(paste0("D:/Data/Coins Data/",pairfolder,"/"))
if(is.na(csvs[1])){
  dir.create(paste0("D:/Data/Coins Data/",pairfolder,"/"))
}csvs_n <- c(as.numeric(gsub("[^[:digit:]]","",csvs)),1)dat<-list()
dat$last <- get_latest(pairfolder,pair)for (i in max(csvs_n):30000){
  if (i==1){

    url <- paste0("[https://api.kraken.com/0/public/Trades?pair=](https://api.kraken.com/0/public/Trades?pair=)",pair,"&since=0")
    dat<-jsonlite::fromJSON(url)$result
    write.csv(dat[pair],paste0("D:/Data/Coins Data/",pairfolder,"/",pair,"_",i,".csv"))
  }
  else{

      dat_last_failsafe <- dat$last
      url<-paste0('[https://api.kraken.com/0/public/Trades?pair=',pair,'&since=',dat$last](https://api.kraken.com/0/public/Trades?pair=',pair,'&since=',dat$last))

      possible_error <-tryCatch(
      dat<-jsonlite::fromJSON(url)$result
      ,error = function(e) e)

      while(is.null(dat$last)){
      Sys.sleep(21)
        print("Wait failsafe")
        url<-paste0('[https://api.kraken.com/0/public/Trades?pair=',pair,'&since=',dat_last_failsafe](https://api.kraken.com/0/public/Trades?pair=',pair,'&since=',dat_last_failsafe))
        dat<-jsonlite::fromJSON(url)$result
      }

      if(!inherits(possible_error,"error")){
      write.csv(dat[pair],paste0("D:/Data/Coins Data/",pairfolder,"/",pair,"_",i,".csv"))
      print(paste("Write",i))
      }
      Sys.sleep(1.05)

  }
}
```

注意:脚本将在提取所有内容后继续运行，因此需要再次检查。提取一枚硬币的所有数据可能需要几个小时。

我摆弄了一下睡眠时间，1 秒多一点似乎没问题。

另一个有用的 API 调用用于检查所有可用于交易和提取的货币对:

```
url <- paste0("[https://api.kraken.com/0/public/AssetPairs](https://api.kraken.com/0/public/AssetPairs)")
dat<-jsonlite::fromJSON(url)$result
save_names <- NULL
for (i in 1:length(dat)){
  name <- dat[[i]]$altname
  if(grepl("EUR",name)){
    save_names <- c(save_names,name)
  }   
}
```

感谢阅读，如果有兴趣，我会在这方面发布更多内容:)