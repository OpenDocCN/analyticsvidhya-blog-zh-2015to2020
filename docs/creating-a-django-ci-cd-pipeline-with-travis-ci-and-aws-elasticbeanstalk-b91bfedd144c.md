# 使用 Travis CI 和 AWS Elasticbeanstalk 创建 Django CI/CD 管道

> 原文：<https://medium.com/analytics-vidhya/creating-a-django-ci-cd-pipeline-with-travis-ci-and-aws-elasticbeanstalk-b91bfedd144c?source=collection_archive---------9----------------------->

本文介绍了使用 Github、Travis CI 和 AWS ElasticBeanstalk 为 Django 应用程序设置 CI/CD 管道的整个过程

## 使用的组件:

*   [Github](https://github.com/) :用于版本控制和代码托管
*   [Travis CI](https://travis-ci.org/) :针对 AWS ElasticBeanstalk 的应用构建和部署
*   [AWS Elasticbeanstalk](https://aws.amazon.com/elasticbeanstalk/) :用于托管 Django 应用程序。