# Watson Personality Insights 简介以及如何在没有 SDK 的情况下访问 Watson

> 原文：<https://medium.com/analytics-vidhya/watson-personality-insights-introduction-and-how-to-access-watson-without-sdk-89eb8992fff2?source=collection_archive---------13----------------------->

![](img/3651dab22e5725bed969dc732b694cb9.png)

马克斯·尼尔森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 双对象。

这篇文章有两个不同的部分:

*   首先，对 Watson Personality Insights 服务进行非技术性介绍。Watson Insights 是 IBM 的一项服务，从社交媒体条目中获取个性资料。
*   第二部分是技术性的，关于如何在没有 SDK 的情况下访问 Watson 服务，直接访问 API 地址。我将解释为什么这很重要。

# 沃森洞察力。

Watson Personality Insights 是 IBM 的一项服务，它根据社交媒体条目，将你的个性分解为五大个性。可怕的一面:为什么人们这么想了解你？答案比问题更简短(也更明显):影响你的决策，并向你出售某些东西。访问链接【https://www.ibm.com/watson/services/personality-insights/ 

五大美食是:

*   对经验的开放:好奇与谨慎。可以分为以下几个方面:冒险性、艺术兴趣、情感性、想象力、智力、自由主义。
*   责任心:有条理与随和。维度:成就努力、谨慎、尽职、有序、自律、自我效能。
*   外向:外向相对保守。方面:活跃程度，自信，快乐，寻求刺激，友好，合群。
*   宜人性:友好与挑战:利他主义，合作，谦虚，道德，同情，信任。
*   神经质:敏感 vs 自信。层面:愤怒，焦虑，抑郁，无节制，自我意识，脆弱。

它也从作者那里得到了需求和价值观。需求清单是:挑战、亲近、好奇、兴奋、和谐、理想、自由、爱、实用、自我表达、稳定、结构。你可以点击此链接阅读这一需求的完整分类:[https://cloud.ibm.com/docs/services/personality-insights?主题=个性-见解-需求](https://cloud.ibm.com/docs/services/personality-insights?topic=personality-insights-needs)

价值观列表是:保守、开放、快乐、自我提升、自我超越。[https://cloud.ibm.com/docs/services/personality-insights?主题=个性-见解-价值观](https://cloud.ibm.com/docs/services/personality-insights?topic=personality-insights-values)

# 附加信息:消费偏好。

URL 中还有一个参数会让营销人员发疯:消费偏好。在这一部分，我们可以将输出细分为以下信息:

*   汽车偏好:所有权和安全性偏好。
*   衣服:对款式、质量、舒适度、品牌影响力的偏好。
*   消费影响来源:效用、在线广告、社交媒体、家庭、冲动购买。
*   其他偏好:健康和活动、环境问题、企业家精神、音乐、阅读和电影。

如果你可能有这种偏好，它会得 1 分，如果没有，它会得 0 分，如果它不能产生这种偏好，它会得 0.5 分:

```
“name”: “Likely to indulge in spur of the moment purchases”,“score”: 0.0},{“consumption_preference_id”: “consumption_preferences_credit_card_payment”,“name”: “Likely to prefer using credit cards for shopping”,“score”: 1.0
```

简而言之:营销活动作者的圣杯。我们幸福吗？继续读下去，很可能你不应该继续读下去。

# “管用吗？”还有一些顾虑。

我没有资格回答这个问题，因为我不是心理学家。我最近检查这项服务是这个测试:我在一个文档中写了一些想法，并上传到 IBM personality Insights。接下来，我做了一个传统的性格测试[https://www.truity.com/test/big-five-personality-test](https://www.truity.com/test/big-five-personality-test)，我将测试结果与沃森的进行了比较。在这两种情况下，大 5 值非常接近。

这不是一个严肃的测试，如果你想要一个更准确的意见，你必须问营销和心理学专家。

另一方面，这个工具可能会引发很多道德和法律问题。好好读一下这些问题:[https://medium . com/Tara az/https-medium-com-Tara az-human-rights-implications-of-IBM-Watsons-personal-insights-942413 e 81117](/taraaz/https-medium-com-taraaz-human-rights-implications-of-ibm-watsons-personality-insights-942413e81117)在这篇文章中，作者谈论了很多关于服务的问题及其背景。很有意思。

# 没有 SDK 的访问。

第一个问题是，什么是 SDK？SDK 是一个额外的模块，IBM 让我们可以更容易地访问它的服务。我们将 SDK 导入到我们的编程语言中，这样我们就可以通过模块进行访问。

第二，为什么我想在没有 SDK 的情况下访问？有两个主要原因:

*   我在微软商务中心工作。这种编程语言不能像 IBM Watson SDK 那样导入模块，所以我必须直接访问 Watson API。我的代码可能对开发环境中不允许使用模块的其他人有用。
*   另一个原因是 IBM 没有为其所有服务提供 SDK。一些测试服务，如自然语言理解，还没有 SDK。

所有的 JavaScript 节点代码都在我的 GIT repo 里:[https://github . com/JalmarazMartn/Watson-personal-insights-node-whit hout-SDK](https://github.com/JalmarazMartn/Watson-personal-insights-node-whithout-SDK)

备注:

```
var request = require(“request”);auth = require(‘./ApiKey.json’);var transUrl = “https://gateway-lon.watsonplatform.net/personality-insights/api/v3/profile?version=2017-10-13&consumption_preferences=true"var data2 = {};var data2 = require(‘./profile.json’);request.post({url: transUrl,
auth,headers:{content_type: ‘application/json’,},body: JSON.stringify(data2)}, function (err, response, body) {console.log(body);});
```

我们向服务发出一个 http 请求，其中包含以下文件:

*   Apikey。是华生的钥匙。我在回购中留了一个例子。
*   侧写。这是包含社交媒体条目的文件。看起来像这样:

```
{“contentItems”: [{“content”: “Trump impeachment conclusion is unpredictable due to lack of antecedents.”,”contenttype”: “text/plain”,”created”: 1447639154000,“id”: “666073008692314113”,”language”: “en”},{“content”: “I have serious doubts about Spain basket team, due important players refusing: Rodr�guez Ibaka Mirotic“contenttype”: “text/plain”,”created”: 1447638226000, “id”: “666069114889179136”,”language”: “en”},{“content”: “Surprising win over Serbia. The keys: defense and Claver performance.”,
```

仅此而已。祝你愉快，小心:有人在看我们(戴上银色的纸帽子以避免它)。