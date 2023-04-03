# 部署拉弗尔 5 号。+基于 Ubuntu 16.04 Apache 的服务器中的项目。

> 原文：<https://medium.com/analytics-vidhya/deploy-laravel-5-4-project-in-ubuntu-16-04-apache-based-server-538d4620bb97?source=collection_archive---------0----------------------->

你好，曾经想知道如何用一个 [apache](https://httpd.apache.org/) 服务器将一个 [***Laravel***](https://laravel.com/docs/5.4/releases) 项目部署到生产环境中吗？也许我想可能有人像我一样是 PHP 框架的初学者。

我不太像一个作家，也不太像一个程序员，所以就让我们深入研究一下吧。

在你动手做这篇文章之前，你需要知道的先决条件就是我想说的:-

1.  [宋承宪](https://www.ssh.com/ssh/protocol/)
2.  简单的终端命令如 *cd* ， *apt-get :*
3.  以下是 Ubuntu 服务器；- [***Apache 服务器***](https://httpd.apache.org/) *安装完毕，PHP 依赖管理工具*[***Composer***](https://getcomposer.org/doc/00-intro.md)*，PHP 7 .**
4.  现有的 Laravel 项目

我们将创建**虚拟主机**，它是一个工具，可以通过[基于名称的](https://httpd.apache.org/docs/2.4/vhosts/name-based.html)或[基于 IP 的](https://httpd.apache.org/docs/2.4/vhosts/ip-based.html)虚拟主机，在一台服务器上运行多个 web 系统。

让我们配置 apache 在虚拟主机中运行我们的 laravel 项目。

通过 scp 命令将您的 laravel 项目上传到***/var/www/html****，即*

> scp-P[端口号]your-laravel 项目的位置**【授权访问服务器的用户】:[您的服务器的 ip 地址]:/var/www/html**

ssh 到您的服务器*，即 ssh-p[端口号][授权访问服务器的用户]:[您的服务器的 ip 地址]*

确保您的源代码和现有软件得到更新。通过运行以下命令

> ***sudo apt-get 更新***
> 
> *后接*
> 
> ***须藤 apt-get 升级***

通过执行以下操作来配置 apache to web 服务器

1.  给予项目目录适当的权限。为此，我们需要允许从 www-data 组访问它，并授予它对存储目录的写权限。

> ***sudo chgrp-R www-data/var/www/html/your-laravel-project
> sudo chmod-R 775/var/www/html/your-laravel-project/storage***

2.通过键入以下命令，为我们的 laravel 项目创建一个配置文件

> ***CD/etc/Apache 2/sites-可用
> 须藤纳米 laravel . con***f

3.在创建的 laravel.conf 文件中，将以下内容粘贴到文件中，保存后关闭。将文件中的[**abedkiloo.com**](http://abedkiloo.com/)替换为你网站的域名，将**abed@abedkiloo.com**替换为等效的电子邮件地址。

> **<虚拟主机*:80 >
> 服务器名 abedkiloo.com**
> 
> **server admin abed@abedkiloo.com
> document root/var/www/html/your-laravel-project/public**
> 
> **<目录/var/www/html/your-laravel-project>
> allow override All
> </目录>**
> 
> **error LOG $ { APACHE _ LOG _ DIR }/error . LOG
> CustomLog $ { APACHE _ LOG _ DIR }/access . LOG 组合
> < /VirtualHost >**

4.启用新创建的。conf 文件并禁用默认值。这是由 apache 预先创建的，并使模式重写能够正常工作。这些是帮助你做到这一点的命令。

> **sudo a2 dissite 000-default . conf***(禁用默认。conf file)*
> **sudo a2 en site laravel . conf***(启用新创建的。conf file)*
> **sudo a2 enmod 重写** *(启用重写)*

5.最后，通过键入以下命令，重新启动 apache 服务器，以便它能够选择您所做的更改

> **sudo 服务 apache2 重启**

6.最后，转到你的 ip 地址或者你在虚拟主机中注册的域名，你的 laravel 项目就可以运行了。

你可能会遇到一些我遇到过的问题，但并不是所有的情况都是这样。这是我遇到的一个主要错误。

> 内部服务器错误
> 
> 服务器遇到内部错误或配置错误，无法完成您的请求。
> 
> 请通过 cbpsmaintenance@uonbi.ac.ke 与服务器管理员联系，告知他们此错误发生的时间，以及您在此错误之前执行的操作。
> 
> 有关此错误的更多信息可以在服务器错误日志中找到。
> 
> cbpsmaintenance.uonbi.ac.ke 端口 80 处的 Apache/2.4.18 (Ubuntu)服务器

这是由于。htaccess 文件在 ***your-laravel-project 的公共目录中不可用。*** *自己创造。htaccess 文件并粘贴此内容，或者从公共目录中的本地项目文件夹上传。htaccess 到您的产品。htaccess*

> **<if module mod _ rewrite . c>
> <if module mod _ negotiation . c>
> Options-MultiViews
> </if module>**
> 
> **重写引擎开启**
> 
> **#如果不是文件夹，重定向尾随斜线……
> rewrite second % {请求文件名}！-d
> 重写者^(.*)/$ /$1 [L，R=301]**
> 
> **#处理前端控制器…
> 重写第% {请求文件名}！-d
> 第二次重写% {请求文件名}！-f
> 重写者 index.php·^【l】**
> 
> **# Handle 授权头
> 重写秒%{HTTP:Authorization}。
> 重写器。*—[E = HTTP _ AUTHORIZATION:% { HTTP:AUTHORIZATION }]
> </if module>**

**注意**你可以创建尽可能多的虚拟主机来服务不同的网页，从一个服务器，但不同的域或 ip 地址使用虚拟主机。

**参考文献**

1.  [数字海洋](https://www.digitalocean.com/community/tutorials/how-to-set-up-apache-virtual-hosts-on-ubuntu-14-04-lts)虚拟主机篇
2.  [叠加过流](https://stackoverflow.com/questions/19071324/request-exceeded-the-limit-of-10-internal-redirects)问题由[最大](https://stackoverflow.com/users/590589/max)
3.  [阿帕奇模式重写文章](http://httpd.apache.org/docs/current/rewrite/intro.html)

谢谢，我希望这能帮助一大堆人。

至少鼓掌吧，伙计