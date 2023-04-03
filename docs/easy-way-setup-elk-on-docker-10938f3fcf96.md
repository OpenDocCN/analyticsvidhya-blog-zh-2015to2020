# 在 Docker 上设置 ELK 的简单方法

> 原文：<https://medium.com/analytics-vidhya/easy-way-setup-elk-on-docker-10938f3fcf96?source=collection_archive---------7----------------------->

人们用麋鹿，传说用码头上的麋鹿。它类似于 Splunk。 **ELK** Stack 旨在允许用户从任何来源以任何格式获取数据，并实时搜索、分析和可视化这些数据。

**第一步:使用 sudo 权限登录&设置弹性搜索的虚拟内存**

`$ sudo sysctl -w vm.max_map_count=262144`

专业提示:要永久设置该值，请更新`/etc/sysctl.conf`中的`vm.max_map_count`设置。