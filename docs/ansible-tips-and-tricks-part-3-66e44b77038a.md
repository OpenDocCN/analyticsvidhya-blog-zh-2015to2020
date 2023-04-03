# 可行的技巧和诀窍(第三部分)

> 原文：<https://medium.com/analytics-vidhya/ansible-tips-and-tricks-part-3-66e44b77038a?source=collection_archive---------4----------------------->

技巧 3:提高执行性能的可行方法

![](img/17a6ff2b007ad501136dc535843e15b9.png)

我们为什么需要这样做？

1.  提高可执行程序的速度。
2.  在更多节点上并行工作。
3.  Ansible 的快速重新运行。
4.  减少 Ansible Server 中的混乱。

让我分享一下我的默认 ansible 配置 aka [**ansible.cfg**](https://github.com/116davinder/kafka-cluster-ansible/blob/master/ansible.cfg) 或者你可以从 GitHub 配置文件[**github.com/116davinder**](https://github.com/116davinder)。

```
[defaults]
host_key_checking = False
command_warnings = False
forks = 100
timeout = 30
retry_files_enabled = False[ssh_connection]
ssh_args=-C -o ControlMaster=auto -o ControlPersist=1200s -o BatchMode=yes
pipelining=False
control_path = /tmp/ansible-%%h-%%p-%%r 
```

L 下面我们来说说上面的配置是如何工作的。

```
host_key_checking = False
```

有了这个答案，忽略 SSH 密钥验证步骤。它确实能在几毫秒内提高速度。

**Ref:**[docs . ansi ble . com/ansi ble/connection _ details . html #主机密钥检查](https://docs.ansible.com/ansible/latest/user_guide/connection_details.html#host-key-checking)

```
forks = 100
```

在 Ansible 中，forks 默认为 5。它允许一次并行播放。

**假设，**
你有 100 台主机，forks = 5，每台主机的任务时间= 2 秒

因为 Ansible 在一批分叉中工作。
**批总数将为** = 100/5 =20
**每批总时间** ~每台主机的任务时间(Ansible 将在该批内的每台主机上并行运行)
**总执行时间将为** = 20 * 2 = 40 秒左右。

现在，如果我们将叉子增加到 50 个或 100 个
，一切都是一样的，但总的批次数量要少得多。
**总批次数** = 100/50 = 2
**总执行时间将为** = 2*2 = 4 秒

> 它将被 **10x** 助推 50 叉。
> 将会是 **20x** 助推 100 叉。
> 超过 100 个叉子就没用了，因为没有更多的主机了。

运行 50 或 100 个分叉的代价是
1。Ansible 上的高 CPU 使用率。
2。网络带宽利用率高。
**参考号:**[docs.ansible.com/ansible/playbooks_strategies.html](https://docs.ansible.com/ansible/latest/user_guide/playbooks_strategies.html)

```
command_warnings = False
retry_files_enabled = False
```

为什么我也使用上面提到的配置，这些与速度优化没有关系，但是它们保持我的系统和 ansible 输出有点干净。
**参:**
1。[docs . ansi ble . com/reference _ appendencies/config . html # command-warnings](https://docs.ansible.com/ansible/latest/reference_appendices/config.html#command-warnings)
2。[docs . ansi ble . com/reference _ appendencies/config . html # retry-files-enabled](https://docs.ansible.com/ansible/latest/reference_appendices/config.html#retry-files-enabled)

```
ssh_args=-C -o ControlMaster=auto -o ControlPersist=1200s -o BatchMode=yes
control_path = /tmp/ansible-%%h-%%p-%%r
```

**SSH 连接优化:**

1.  使用 **ControlPersist** 增加连接时间长度。
2.  **BatchMode** (禁用基于 SSH 的交互提示)。
3.  **控制路径** SSH 临时套接字路径并保持每个主机的唯一模式(ansible-%%h-%%p-%%r)，这将允许 ansible 在需要时共享连接，并在必要时重用。

**Ref:**
1。[https://en.wikibooks.org/wiki/OpenSSH/Cookbook/Multiplexing](https://en.wikibooks.org/wiki/OpenSSH/Cookbook/Multiplexing)2。[https://www . techrepublic . com/article/how-to-use-multiplexing-to-speed-up-the-ssh/](https://www.techrepublic.com/article/how-to-use-multiplexing-to-speed-up-the-ssh/)
3 .[https://linux.die.net/man/5/ssh_config](https://linux.die.net/man/5/ssh_config)(batch mode)
4。[docs . ansi ble . com/plugins/connection/ssh . html # parameter-control _ path](https://docs.ansible.com/ansible/latest/plugins/connection/ssh.html#parameter-control_path)

```
pipelining=False
```

应该禁用它，因为它与 become 冲突，并且要求您在目标服务器上禁用“requiretty”。

**Ref:**
1。[docs . ansi ble . com/plugins/connection/ssh . html #参数管道](https://docs.ansible.com/ansible/latest/plugins/connection/ssh.html#parameter-pipelining)

上述配置可以在可行的执行中节省大量时间。请记住，上述配置可以在一定程度上节省您的时间，您必须将您的剧本/角色设计为最佳性能。

如果不需要，不要使用“串行”模式或[线性策略](https://docs.ansible.com/ansible/latest/plugins/strategy/linear.html#linear-strategy)。

希望它能节省你的时间。快乐编码又名 Ansible 编码。