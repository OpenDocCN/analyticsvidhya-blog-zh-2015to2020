# 冠状动脉疾病预测

> 原文：<https://medium.com/analytics-vidhya/coronary-artery-disease-prediction-68a681bd7002?source=collection_archive---------6----------------------->

## 使用集合方法和推进技术

![](img/189fbacdd225d50a7c8f75bde5dafc27.png)

来自[疾病预防控制中心](https://www.cdc.gov/heartdisease/coronary_ad.htm)站点的图片

你听说过机器预测一个人的疾病吗？嗯，如果你有正确的数据集，那么就有可能建立一个智能系统，它可以预测疾病，甚至不需要人们通过侵入性疾病识别程序。

在这篇博客中，我将带你浏览预测一个人冠状动脉阻塞的机器学习模型。我所使用的数据集可在 [**UCI 机器学习库**](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/) 上获得，并由以下机构捐赠:

```
**1.** Hungarian Institute of Cardiology. Budapest: Andras Janosi,M.D.**2.** University Hospital, Zurich, Switzerland: William Steinbrunn,M.D.**3.** University Hospital, Basel, Switzerland: Matthias Pfisterer,M.D.**4.** V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.
```

# 下载数据集

我们将使用**克利夫兰**数据集，因此从 UCI ML 资源库链接下载文件 **processed.cleveland.data** 。文件 **heart-disease.names** 中还提供了数据集特征的简要信息。

我还创建了一个单独的文档( [***特性详解 handbook.docx***](https://github.com/Rajesh-ML-Engg/Coronary_Artery_Disease/blob/master/Features%20detailed%20handbook.docx) 上传到[***GitHub***](https://github.com/Rajesh-ML-Engg/Coronary_Artery_Disease))供您参考，以深入了解数据集特性。

![](img/fe07218877a91a92147a0e46dea118b6.png)

# 什么是冠状动脉疾病？

冠状动脉疾病(CAD)是最常见的心脏病类型，当为你的心脏提供血液、氧气和营养的主要血管(**冠状动脉**动脉)受损或患病时，就会发生这种疾病。也就是俗称的冠心病或缺血性心脏病。

# 什么导致冠状动脉疾病？

动脉中含胆固醇的沉积物(斑块)和炎症通常是冠状动脉疾病的罪魁祸首。

斑块是由动脉中的胆固醇和其他物质沉积而成的。随着时间的推移，斑块的堆积会导致动脉内部变窄，这可能会部分或阻碍血液流动。

# 导入克利夫兰数据集

![](img/8d370afe4bb0e361f8cf3821d427ba29.png)

导入了克利夫兰数据集

![](img/b128c0a59639f20ace1e3e029e3b8793.png)

根据数据集为要素指定名称

# 数据预处理

这是我们在创建任何 ML 模型之前执行的第一步。在此步骤中，我们执行各种活动，如数据类型处理、填充缺失值、要素缩放、要素变换、异常值处理等，以清理数据集并移除不需要的噪声。

## 步骤 1:在“num”中处理类

处理多变量预测属性，即' **num** '，其中值为> 1 的记录受 CAD 影响，< 1 为非 CAD。

**在应用任何操作之前检查计数**

![](img/f3de160c0ba4e621ca9d89ca79b0c05e.png)

步骤 1.1

**查找堵塞记录的数量> 50%**

![](img/cba4158d19a488c7689bb9092f81bd39.png)

步骤 1.2

**将“数字”特征分为两类 0(堵塞< 50%)和 1(堵塞> 50%)**

![](img/3af46e11aefa5067059455e8f6ef0622.png)

步骤 1.3

## 可视化 CAD 和非 CAD 记录

![](img/6b318f059d7aa9fbbb46b6ffd129ee6e.png)

## 发现缺少值('？'))在数据集中

![](img/a3f0e1e2aa580bb1cc0a1dbd8ce331cf.png)

找出要素中缺失值的计数

## 可视化丢失的记录

![](img/bdfceae031ee9cebef86c1cdd8fc30cb.png)

数据集要素中缺少值可视化

## 具有空颜色 _vsl 的记录

![](img/2b5c0395b577015e5a2f961716c59120.png)

## 填充彩色 VSL 特征中缺失的值

![](img/2c307481dd50467082e2f0f06dbcb4dc.png)

修复了彩色 VSL 中缺失的值

## 填充 THAL 特征中缺失的值

![](img/5e6e5db0266a3c84ff4e47ec766e5b27.png)

修复了 THAL 中缺失的值

## 再次可视化丢失的值

![](img/9087cd0214588c99b23876f66d0f017b.png)

这一次，我们得到了清晰的热图，任何特性都没有缺失值。

## 数据类型处理

![](img/68d33a97c6e9acf368f23f099848da80.png)

列数据类型

![](img/9fea8e1e30d57942eb4ee5255ccb0587.png)

UDF 更改列数据类型

## 删除不需要的列

![](img/69af525efb3fd7db28226f32b5ddae6a.png)

因为我们已经为 COLOR_VSL、THAL 和 NUM 创建了更新的列，因此删除了它们的早期版本。

# 探索性数据分析

EDA 是一种主要使用可视化方法来分析数据集并总结其主要特征的方法。对数据集执行 EDA 的最佳方式是向自己提问，并尝试找到答案。EDA 的主要目标是启用数据，以便它可以告诉你超越正式建模或假设测试的见解。

## 问题 1:29-48 岁年龄组中有多少人阻塞率超过 50%？

![](img/8aba28676c5d708a61729f764d6d106d.png)

## 问题 2:48-56 岁年龄组中有多少人阻塞率超过 50%？

![](img/85b0796aa0344f6742e5eddc90a50fe6.png)

## 问题 3:56-77 岁年龄组中有多少人阻塞率超过 50%？

![](img/0e7df9772c2e0124605885305effbd6a.png)

## 问题 4:多少男性和女性有心脏病？

![](img/3a1843d64bd1f83f81f22ca349b3bb9b.png)

## 问题 5:有多少病人患有各种胸痛？

![](img/07ed9475601c2fdb67868ecc9903aadc.png)

## 问题 6:静息状态下的高血压是否对应于冠心病？

![](img/7762e800302a83475fbe882491319e19.png)

**BP 组 1:[94–120]**

![](img/8516d04e62b2daa984904dccbe69e630.png)

**BP 组 2:[120–130]**

![](img/b33e9fab105545345757e40673120585.png)

**BP 组 3:[130–140]**

![](img/350b35a309aa2fc6b74085becc72ee45.png)

**血压组别 4: 140 或以上**

![](img/221146c3f9c522445679b35d9ac028df.png)

## 问题 7:高血压是否与高血清胆固醇相对应，也会导致冠心病？

![](img/7dc6c4895a80b2fab3de0bfaf6814076.png)![](img/3332365125df0b04fa9ecd58ec0f3d9d.png)

## 问题 8:高血压是否对应高血清胆固醇，并导致高血糖？

![](img/cd1c92856de2aca317188998672f9ec7.png)![](img/1a07ff4dc5ccafeb0684fbfd6aecb8a3.png)

## 问题 9:高血压是否与高血糖相对应，也会导致冠心病？

![](img/c8bdd0194c8edf58325fdfcf8728e3a6.png)

## 问题 10:ST 段异常对应导致冠心病吗？

![](img/1eb09d9530e7fdb88432c93b98d89f93.png)

解决方案-10.1

![](img/14b1f125a5f78db98310dcd5afd1873d.png)

解决方案-10.2

## 问题 11:左心室肥厚与血压和胆固醇有关吗？

![](img/88a4ed59d4e7f621b91d7b01ec4740dc.png)

解决方案-11.1

![](img/6e7139c1b1a45c93387f9c855ced0180.png)

解决方案-11.2

## 问题 12:左心室肥厚是否与高血糖有关，也会导致冠心病？

![](img/0d37075d5e82a2c8e830697fd5a9a20f.png)

# 问题 13:最大心率对应于静止时的血压吗，也会导致冠心病吗？

![](img/60cc6ff4959c4f9bfd21cfe5b5a12cc6.png)

解决方案-13.1

![](img/4bcaa5b0225d15fc8bbb6c91db07bcf4.png)

解决方案-13.2

## 问题 14:运动诱发的心绞痛与冠心病对应吗？

![](img/3ad735370cc81a2c87b2adfce0a071f4.png)

## 问题 15:运动诱发的心绞痛和老年峰如何与冠心病结果相对应？

![](img/5982580e20b3360b0597d0fc260a489b.png)

## 问题-16:运动试验中什么样的 ST 斜率更对应 CAD？

![](img/ba78603b3210c7467c49eb50cf8bf54b.png)

## 问题 17:ST 段斜率与旧峰值和最大心率有关系吗？

![](img/2b1bc8c64abfb39c99ae02aba77c7ccf.png)

解决方案-17.1

![](img/823350be0f4c06ac3bdb535af74887e3.png)

解决方案-17.2

## 问题-18:地中海贫血与年龄或最大心率/血压/胆固醇有关系吗？

![](img/ad992384d6038a13208537dddeff4910.png)

## 处理分类变量

![](img/09ed8d3d461d451d9e66897b415a77fe.png)

使用 **get_dummies** 处理分类变量

## 数据分布和异常值检测图

![](img/e57db430477db00276137b271ee5cc27.png)

**功能:REST_BP**

![](img/ae2d265e6735279daf09260c2d16da9e.png)

**功能:REST_BP**

![](img/c08285ed87238e57d46e0fe1b54d23ef.png)![](img/0b378281e5b062dd76e33c1e5dde3e23.png)![](img/2e13a61cdf8a8d3308afecd652ccf9e4.png)![](img/e9d88c70e77dd5d9a9c0520650e67784.png)![](img/959641b3310087722636beb2591b8a97.png)![](img/c06284eed66281e2b5ac0f0dd1c9f11e.png)![](img/ee105bb959f0ba1b3f7892f6017979ad.png)![](img/2044c1ad6d8b3286a2e6aec553289678.png)

## 特征缩放和变换

![](img/5abeed1d15c8f576f9d4cc69ae95cf49.png)

## 定量(原始、缩放和转换)特征中的异常值可视化

![](img/e2bb3b02df149b289dbc5057122ef3ff.png)

原始要素和缩放要素中的异常值

![](img/80072f20ef616d6daa110ee8afef344f.png)

变换后的要素中没有异常值

# 特征相关图

![](img/0bdc90f875ef5fca2c7a548a0f6c3991.png)

# 模型选择和评估

![](img/dd5d294a6e349424f12bd23802607557.png)

**分叉特征和标签**

## 导入所需的包并执行数据过采样

![](img/f9f066faaa0c22570f634532a0fb63e7.png)![](img/41554f82454180856d4f7e4526fca3f9.png)

**朴素贝叶斯**

![](img/e464ebed60380b4bd048ff6652b6b1dc.png)

**梯度提升分类器**

![](img/f0f004391838095aad71e93f052312a5.png)![](img/166550bb3a39af464ae08f20c92aec4a.png)![](img/eb27d1e2a1adc7e0d9f6ea21834fac82.png)

# 模型超参数化/调整

![](img/e663ed04654cf94f8f86dfc50a61c935.png)

**学习率和 N 估计量**

![](img/e44944a98e81e642d04158ef085e618c.png)

**最大深度**

![](img/daaf3f48cff110540fde694d4ac0a4a9.png)

**最小样本 _ 叶和最小样本 _ 分割**

![](img/3a601e664a8ba6568d2aa3b06b75688a.png)

**MAX_FEATURES**

![](img/b1ad189aad5225d531df19dcd412981f.png)

**子样本**

## 超参数化后的结果

![](img/33abef52996569ec52b3072675bd8837.png)

恭喜你，我们已经到了这个博客的结尾。总之，我们使用心脏病数据集来确定对一个人的心脏病进行分类的最佳分类算法。此外，我使用了各种机器学习模型来最准确地预测结果。

> **如果你想下载这个博客的 Jupyter 笔记本，请访问下面的 GitHub 库:**
> 
> [**https://github.com/Rajesh-ML-Engg/Coronary_Artery_Disease**](https://github.com/Rajesh-ML-Engg/Coronary_Artery_Disease)

谢谢你，祝你学习愉快！！！！