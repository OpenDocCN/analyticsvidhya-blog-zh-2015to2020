# Python 中随机森林的留一人交叉验证分步指南

> 原文：<https://medium.com/analytics-vidhya/step-by-step-guide-to-leave-one-person-out-cross-validation-with-random-forests-in-python-34b2eaefb628?source=collection_archive---------2----------------------->

我已经收到了许多关于如何在随机森林中实现留一个人在外交叉验证的请求。我将我的方法提炼成一个函数，并发表在[数字生物标记发现管道(DBDP)](http://dbdp.org) 上。如果您想在自己的随机森林工作中实现这种交叉验证，请访问 DBDP GitHub 上的[库。使用该功能的“操作方法”有很好的文档记录，在许多项目中只需**一行**就能轻松实现。](https://github.com/Big-Ideas-Lab/DBDP/tree/master/DigitalBiomarkers-generalML/loocvRF)