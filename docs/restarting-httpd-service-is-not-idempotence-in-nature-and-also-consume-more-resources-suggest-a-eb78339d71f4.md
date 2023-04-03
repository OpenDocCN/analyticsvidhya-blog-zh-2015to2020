# 重启 HTTPD 服务本质上不是幂等的，并且还消耗更多的资源，建议在 Ansible Playbook 中纠正这个挑战的方法

> 原文：<https://medium.com/analytics-vidhya/restarting-httpd-service-is-not-idempotence-in-nature-and-also-consume-more-resources-suggest-a-eb78339d71f4?source=collection_archive---------17----------------------->

![](img/8213b50f36025874cb8e154b96adc70f.png)

**问题陈述:**

可互换性质遵循幂等性，但是一些模块和关键字不遵循幂等性。在这篇博客中，我们将解决一个问题，其中一个模块关键字没有遵循幂等，即**重启** HTTPD 服务。

**对于上面的问题陈述，我有两个解决方案。我将借助条件给出第一个解。**

**1 >第一法**

```
- hosts: all
  tasks:- file:
     state: directory
     path: "/mount"- mount:
     src: "/dev/cdrom"
     path: "/mount"
     state: mounted
     fstype: "iso9660"- yum_repository:
     baseurl: "/mount/AppStream"
     name: "RepO1"
     description: "This is the configuration for the Yum !"
     gpgcheck: no- yum_repository:
     baseurl: "/mount/BaseOS"
     name: "RepO2"
     description: "This is the configuration for the Yum !"
     gpgcheck: no

  - package:
     name: "httpd"
     state: present

  - file:
     state: directory
     path: "/var/www/NewDir"- copy:
     dest: "/etc/httpd/conf.d/new.conf"
     src: "new.conf"
    register: info    

  - copy:
     dest: "/var/www/NewDir/index.html"
     content: "Welcome TO Coding Pathshala \n"- name: "Normally Starting the Service of httpd"
    service:
     name: "httpd"
     state: started
     enabled: yeswhen: info.changed == false- firewalld:
     port: 8080/tcp
     state: enabled
     immediate: yes- name: "httpd_service_restart"
    service:
       name: "httpd"
       state: "restarted"
    when: info.changed == true
```

因此，在**寄存器模块**的帮助下，我将该模块的元数据值存储在一个**变量**中。我使用的条件是值以映射的形式存储。它由一个名为 changed 的**键组成。如果**键的值为 false** ，则**意味着模块没有进行任何更改**，如果 key 的**值为 true，则模块已经进行了更改**。**

我们将使用 r **注册模块**和**复制模块**，在这里我们可以更改文件*的**配置(当我们在配置文件中进行一些更改时，我们只需要重新启动服务)*** 。

注册模块将帮助我们在变量中存储复制模块的元数据。该值以映射(键值对)的形式存储。我们将使用*这个变量两次。*

首先，当我们在关键字 **started** 的帮助下**启动 web 服务器**以及在关键字**restart**的帮助下启动服务时，我们使用这个变量。

started 关键字只有在条件存在时才会运行

> 变量名称.已更改==false

重启的模块将在条件存在时运行，即

> 变量名称.已更改= =真

**2 >第二种方法**

第二个完美的解决方案是，当配置文件发生变化时，只在重启时运行服务，也就是说，如果配置文件没有变化，就不要重启(始终遵循幂等概念)。

这就是所谓的 Notifies ( **notify** 也是 Ansible 中存在的模块)。因此，我们需要将这个模块放在 Handler 下的重启服务任务中..

因此，如果这个模块通知在配置中有一些变化，那么由处理程序调用的重新启动的服务..

```
- hosts: all
  tasks:- file:
     state: directory
     path: "/appdvd"- mount:
     src: "/dev/cdrom"
     path: "/mount"
     state: mounted
     fstype: "iso9660"- yum_repository:
     baseurl: "/mount/AppStream"
     name: "RepO1"
     description: "This is the configuration for the Yum !"
     gpgcheck: no- yum_repository:
     baseurl: "/mount/BaseOS"
     name: "RepO2"
     description: "This is the configuration for the Yum !"
     gpgcheck: no

  - package:
     name: "httpd"
     state: present

  - file:
     state: directory
     path: "/var/www/NewDir"- copy:
     dest: "/etc/httpd/conf.d/new.conf"
     src: "new.conf"
    notify: httpd_service_restart    

  - copy:
     dest: "/var/www/NewDir/index.html"
     content: "Welcome TO Coding Pathshala \n"- service:
     name: "httpd"
     state: started
     enabled: yes

  - firewalld:
     port: 8080/tcp
     state: enabled
     immediate: yeshandlers:
  - name: httpd_service_restart
    service:
       name: "httpd"
       state: "restarted"
```

感谢阅读这篇博客！！

想要下载行动手册，然后前往 github →

[https://github.com/Shashwatsingh22/Ansible](https://github.com/Shashwatsingh22/Ansible)