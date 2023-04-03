# Firebase Web 推送通知

> 原文：<https://medium.com/analytics-vidhya/firebase-web-push-notifications-26352b136814?source=collection_archive---------0----------------------->

本文将帮助您实现 firebase web 推送通知，而无需在项目中使用任何最新版本 7.18.0 或更高版本的包。

![](img/07aaf47fe93e2641b3d162bca2984f10.png)

# **简介**

[Firebase](https://firebase.google.com/) 使用云服务在 Android、IOS &网络上提供通知服务。Firebase 云消息或 FCM 基于**令牌**的基本原理运行，该令牌是为每个设备&唯一生成的，稍后用于发送…