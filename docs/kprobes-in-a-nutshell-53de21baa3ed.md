# 简而言之

> 原文：<https://medium.com/analytics-vidhya/kprobes-in-a-nutshell-53de21baa3ed?source=collection_archive---------33----------------------->

【https://ish-ar.io/kprobes-in-a-nutshell/】原载于

# Kprobes 是什么？

![](img/63071faf3e0479e2d99ba871ed73f860.png)

下面是来自 kernel.org 的定义:

*"Kprobes 使您能够动态地闯入任何内核例程，并无中断地收集调试和性能信息。您可以在几乎任何内核代码地址[1]进行陷阱，指定一个当断点被命中时调用的处理程序例程…*

*…当 CPU 命中断点指令时，会发生一个陷阱，CPU 的寄存器被保存，控制通过 notifiercallchain 机制传递给 Kprobes。Kprobes 执行与 kprobe 相关联的“pre_handler ”,将 kprobe 结构和保存的寄存器的地址传递给处理程序。”*

~所以基本上它允许你运行两个函数，pre *处理程序和 post* 处理程序，每次被探测的函数被调用~

说实话，第一次听说 Kprobes、Jprobes、Kretprobes 等等……对我来说都有点复杂。很高兴地说，经过几个小时的研究，现在开始有意义了。

请注意，现在有一种比我今天向您展示的方法更简单的方法来使用 Kprobes 但是我将在下一篇文章中介绍这种方法。没错，我说的就是 bpf()！

**那么我们今天将如何使用 Kprobes 呢？**

轻松点。通过创建一个简单的内核模块，将它插入我们的内核并测试它。不要害怕，这是一个非常简单的任务，即使听起来很棘手。

**教程目标**:创建一个内核模块，每当函数${function}被使用时，它就使用 Kprobes 进行计数。

先说第一件事:需求！

你需要一台 **Linux 机器**！

![](img/9cf1ae13ed244d1c74d984959cd340b8.png)

*注意:我只在我的私人服务器(Ubuntu 18.04.2 LTS 仿生海狸)上测试了这个过程，所以如果你使用不同的操作系统，你可能需要找到正确的软件包名称，我们将创建的内核模块可能无法在不同的架构上工作。*

1.  创建工作目录 dir 并安装所需的软件包。

```
mkdir ./ish-ar.io-lab/ && \
touch ./ish-ar.io-lab/{Makefile,ish.c} && \
cd ./ish-ar.io-lab/apt-get update && \
apt-get install gcc strace make libelf-dev -y
```

2.如下编辑文件`Makefile`:

```
obj-m +=ish.o
KDIR= /lib/modules/$(shell uname -r)/build
all:
		$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules
clean:
		rm -rf *.o *.ko *.mod.* .c* .t*
```

*注意:当你需要在* `*Makefile*` *中调用* `*make*` *时，最好使用变量* `*$(MAKE)*` *而不是命令。*

**重要提示:确保你在 Makefile 中使用的是制表符而不是空格，否则你会得到一个错误提示:**

```
Makefile:N: *** missing separator (did you mean TAB instead of 4 spaces?).  Stop.
```

3.我们需要找出我们想要计数/截取的函数。

在这个例子中，我想计算每次程序执行的次数。所以我搜索了我想要的函数，就像这样:

```
strace ls 2>&1 | less
```

在顶部，您应该会看到类似这样的内容:

```
execve("/bin/ls", ["ls"], 0x7fff38f23780 /* 21 vars */) = 0
```

看起来`execve`就是我们要拦截的函数！我们现在需要它的内存地址来探测它。所以让我们搜索一下:

```
root@ip-172-31-3-95:~/lab# grep sys_execve /proc/kallsyms
ffffffffbcc7f010 T sys_execve
ffffffffbcc7f050 T sys_execveat
ffffffffbcc7f0b0 T compat_sys_execve
ffffffffbcc7f100 T compat_sys_execveat
```

*如果你不知道这个文件* `*/proc/kallsyms*` *是什么，你可以看看这个页面->*[*https://onebitbug . me/2011/03/04/introducing-Linux-kernel-symbols/*](https://onebitbug.me/2011/03/04/introducing-linux-kernel-symbols/)

因此，我们有一个名为 sys_execve 的函数，它的地址是 ffffffffbcc7f010。

4.现在编辑文件`ish.c`:

我们需要包含所需的库，所以在 C 程序的顶部键入:

*注意:这个库# include<Linux/kprobes . h>正如你从它的名字可以注意到的，它是使用 k probes 的基础。*

```
#include<linux/module.h>
#include<linux/version.h>
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/kprobes.h>
```

在 includes 之后，创建两个简单的结构。我们以后会需要它们。

```
static unsigned int counter = 0;
static struct kprobe kp;
```

你还记得我曾经写过前置*处理程序和后置*处理程序的功能吗？让我们先创建它们。

*提醒一下:prehandler 函数在我们截取的函数之前执行，posthandler 函数在它之后执行。*

```
int kpb_pre(struct kprobe *p, struct pt_regs *regs){
 printk("ish-ar.io pre_handler: counter=%u\n",counter++);
 return 0;
}void kpb_post(struct kprobe *p, struct pt_regs *regs, unsigned long flags){
 printk("ish-ar.io post_handler: counter=%u\n",counter++);
}
```

在这两个函数之后，让我们创建我们的模块入口点和出口点。

```
int minit(void)
{
 printk("Module inserted\n ");
 kp.pre_handler = kpb_pre;
 kp.post_handler = kpb_post;
 kp.addr = (kprobe_opcode_t *)0xffffffff8d67f010;
 register_kprobe(&kp);
 return 0;
}void mexit(void)
{
 unregister_kprobe(&kp);
 printk("Module removed\n ");
}
module_init(minit);
module_exit(mexit);
MODULE_AUTHOR("Isham J. Araia");
MODULE_DESCRIPTION("[https://ish-ar.io/](https://ish-ar.io/)");
MODULE_LICENSE("GPL");int minit(void) {  printk("Module inserted\n ");  kp.pre_handler = kpb_pre;  kp.post_handler = kpb_post;  kp.addr = (kprobe_opcode_t *)0xffffffff8d67f010;  register_kprobe(&kp);  return 0; }  void mexit(void) {  unregister_kprobe(&kp);  printk("Module removed\n "); } module_init(minit); module_exit(mexit); MODULE_AUTHOR("Isham J. Araia"); MODULE_DESCRIPTION("https://ish-ar.io/"); MODULE_LICENSE("GPL");
```

每次你插入这个模块时，函数 minit 将被触发，如果你移除内核模块，函数 mexit 将被调用。

**重要**:将`kp.addr = (kprobe_opcode_t *)0xffffffff8d67f010;`替换为您在第 3 步发现的函数内存地址— > `kp.addr = (kprobe_opcode_t *)0xFUNCTION_MEMORY_ADDRESS;`。

5.您早期创建的内核模块应该如下所示:

```
#include<linux/module.h>
#include<linux/version.h>
#include<linux/kernel.h>
#include<linux/init.h>
#include<linux/kprobes.h>static unsigned int counter = 0;static struct kprobe kp;int kpb_pre(struct kprobe *p, struct pt_regs *regs){
    printk("ish-ar.io pre_handler: counter=%u\n",counter++);
    return 0;
}void kpb_post(struct kprobe *p, struct pt_regs *regs, unsigned long flags){
    printk("ish-ar.io post_handler: counter=%u\n",counter++);
}int minit(void)
{
    printk("Module inserted\n ");
    kp.pre_handler = kpb_pre;
    kp.post_handler = kpb_post;
    kp.addr = (kprobe_opcode_t *)0xFUNCTION_MEMORY_ADDRESS;
    register_kprobe(&kp);
    return 0;
}void mexit(void)
{
    unregister_kprobe(&kp);
    printk("Module removed\n ");
}module_init(minit);
module_exit(mexit);
MODULE_AUTHOR("Isham J. Araia");
MODULE_DESCRIPTION("[https://ish-ar.io/](https://ish-ar.io/)");
MODULE_LICENSE("GPL");
```

6.现在让我们构建并插入我们的模块。

在您的工作目录中键入:

```
make
```

您应该得到如下输出:

```
make -C /lib/modules/4.15.0-1044-aws/build SUBDIRS=/root/ish-ar.io-lab modules
make[1]: Entering directory '/usr/src/linux-headers-4.15.0-1044-aws'
CC [M]  /root/ish-ar.io-lab/ish.o
Building modules, stage 2.
MODPOST 1 modules
CC      /root/ish-ar.io-lab/ish.mod.o
LD [M]  /root/ish-ar.io-lab/ish.ko
make[1]: Leaving directory '/usr/src/linux-headers-4.15.0-1044-aws'
```

要插入模块类型:

```
insmod ish.ko
```

要查看模块是否已加载，请键入:

```
root@ip-172-31-3-95:~/ish-ar.io-lab# lsmod | grep ish ish                    16384  0
```

有用吗？来测试一下吧！
我们需要执行一些东西，所以让我们键入`ls`，然后看到 dmesg:

```
root@ip-172-31-3-95:~/ish-ar.io-lab# dmesg [ 4813.434548] Module inserted 
[ 4815.142934] ish-ar.io pre_handler: counter=0
[ 4815.142935] ish-ar.io post_handler: counter=1
```

所以如果你有这样的输出…是的！有用！

要删除该模块，只需键入:

```
rmmod ish
```

**重述:**

我们学到了什么？

**如何使用内核模块使用 Kprobes，pre *和 post* 处理程序是什么，如何使用它们在每次调用函数时进行计数(例如:sys_execve)**