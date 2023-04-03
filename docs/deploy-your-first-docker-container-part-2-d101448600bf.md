# 部署您的第一个 Docker 容器—第 2 部分

> 原文：<https://medium.com/analytics-vidhya/deploy-your-first-docker-container-part-2-d101448600bf?source=collection_archive---------20----------------------->

在上一篇文章中，我们学习了如何搜索和运行特定的部署映像。现在，我们将进一步了解部署您第一个 docker 容器的步骤:

**第二步:寻找运行中的容器**

启动的容器在后台运行，docker ps 命令列出了所有正在运行的容器、用于启动容器的映像和正常运行时间。

该命令还显示友好的名称和 ID，可用于查找有关各个容器的信息。

命令

> 码头工人检查<friendly-name></friendly-name>

提供有关正在运行的容器的更多详细信息，如 IP 地址。

命令

> 码头日志<friendly-name></friendly-name>

将显示容器已写入标准错误或标准输出的消息。

**第三步:访问 Redis**

Jane 很高兴 Redis 正在运行，但是很惊讶她不能访问它。原因是每个容器都是沙箱化的。如果服务需要由不在容器中运行的进程访问，那么端口需要通过主机公开。

一旦公开，就可以像在主机操作系统本身上运行一样访问该进程。

Jane 知道，默认情况下， *Redis* 运行在端口 *6379* 上。她已经了解到，默认情况下，其他应用程序和库期望一个 *Redis* 实例在端口上侦听。

阅读文档后，Jane 发现当使用 *-p <主机端口> : <容器端口>* 选项启动容器时，端口被绑定。Jane 还发现在启动容器时定义一个名称很有用，这意味着她不必使用 Bash 管道或者在试图访问日志时一直查找名称。

Jane 找到了在后台运行 *Redis* 的最佳方法，在端口 *6379* 上运行一个名为 *redisHostPort* 的端口，使用下面的命令`docker run -d --name redisHostPort -p 6379:6379 redis:latest`

## 步骤 4:端口映射

在固定端口上运行进程的问题是，您只能运行一个实例。Jane 更喜欢运行多个 Redis 实例，并根据 Redis 运行的端口来配置应用程序。

经过试验，Jane 发现仅仅使用选项 *-p 6379* 就能使她暴露 *Redis* 但是是在一个随机可用的端口上。她决定用`docker run -d --name redisDynamic -p 6379 redis:latest`来测试她的理论

虽然这种方法有效，但她现在不知道分配了哪个端口。谢天谢地，这是通过`docker port redisDynamic 6379`发现的

Jane 还发现，列出容器会显示端口映射信息，`docker ps`

**下一步，我们将继续第三部分**