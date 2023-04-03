# 使用 Python 下载和重采样加密交易数据

> 原文：<https://medium.com/analytics-vidhya/downloading-and-resampling-crypto-trade-data-with-python-5059ddcc9bcc?source=collection_archive---------6----------------------->

![](img/b32b82708a2ac09355d0f21ee9849a34.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上由 [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) 拍摄的照片

本文将带您了解如何获取给定加密货币的交易数据(我们将使用北海巨妖交易所 API)并对其进行重采样，以获取所需频率的 OHLC 数据。

**简介**

近年来，电子货币交易获得了极大的欢迎。构思有利可图的交易策略需要深思熟虑…