# Python 3 迁移——宝贵的经验

> 原文：<https://medium.com/analytics-vidhya/python-3-migration-valuable-lessons-efd0d0061609?source=collection_archive---------17----------------------->

我最近得到一个机会，通过移动应用程序，迁移分布在面向消费者的应用程序的多个回购中的 26K 行代码，为数百万客户提供服务。这是运行在 gevent 上的 flask 应用程序，在 AWS ECS 中有许多后台任务。我觉得这次经历对社区很有帮助。我试着把学到的东西写在这里。

注意:您可以在上找到格式更好的版本

idea—[https://www . idea . so/Python-3-Migration-valued-Lessons-78 abf 495 E1 e 4486 a 81116 c 98 D2 C3 fc 66](https://www.notion.so/Python-3-Migration-Valuable-Lessons-78abf495e1e4486a81116c98d2c3fc66)

[https://gist . github . com/mthipparthi/2 e5b 9 B4 e 224390959 f 6143 e 9 BD 94 E1 e 7](https://gist.github.com/mthipparthi/2e5b9b4e224390959f6143e9bd94e1e7)

# 测试

迁移首先需要的是——单元测试的存在。如果根本没有单元测试，请首先编写它们。如果你有单元测试，并且你觉得它们不够充分，首先花时间写那些单元测试用例。在没有测试的情况下，不要尝试迁移。

智慧

*   创造。用 Py2 和 Py3 做所有的测试。这节省了很多时间
*   创建`locust`测试(负载测试)来测试`uat`中的端到端流程。

# 分支

在现实世界中，特性添加将优先于迁移。因此，当您为迁移创建一个分支时，这将比您想象的持续时间更长。为大量合并冲突做好准备。

# 承诺

在逻辑上解决一个问题时，创建尽可能多的提交。一旦你解决了一些逻辑问题，提交它并创建尽可能多的提交，当你将来需要引用时，它们会有很大的帮助。当您有多个 repos 要迁移时，您最终会引用它们。

# 冲突

如果这个项目是长期运行的，请试着用你的`develop/staging`重新调整几个星期。如果你有很长一段时间没有更新，你会有很多惊喜。

智慧

*   这个命令`git config --global rerere.enabled true`会有很大的帮助，因为它会记住你做的所有解决方案，并在随后的冲突中重放。它节省了很多时间。
*   有时你会忘记解析一些(非 python 文件，隐藏的(circleci config))，这会导致以后的混乱。为了避免这个命令会有很大的帮助`ag ">>>>>" --hidden --ignore-dir .git`总是在你做之前执行它`git add .;git rebase --continue`

# 代码审查

如果值得一次或在提交级别审查所有代码，请在团队中工作。拥有尽可能多的提交肯定会对评审者有所帮助。只要单元测试通过，这就是代码工作的良好迹象。你可以在你的团队中决定什么是正确的策略。因为变更很难审核。如果你删除了一些代码或者测试用例或者库，让你的团队保持循环。

# 生产部署。

# 金丝雀

除了在低影响的情况下，我们应该结束并行运行 Py2 和 Py3 应用程序，直到 Py3 为您的所有流量提供服务。根据您的设置，如果您的应用面向 CDN，则很容易将流量转移到两个应用，否则您需要使用自己版本的 canary 进行迁移，而不会出现任何停机。使用共享资源运行 Py2 和 Py3 有其自身的挑战。

如果您使用 ECS，您可以为 Py2 和 Py3 创建两个单独的服务，并相应地转移流量。

# 禁止共享腌制物品。

在迁移中，缓存是您的敌人。我得到了惨痛的教训。Py2 和 Py3 更是如此，它们将对象缓存为 pickle 对象。Python 2 使用协议 0–2 来挑选对象，而 Python3 使用协议 3–5。如果您通过任何缓存(redis)共享这些对象，它们是不兼容的，并且应用程序会在生产中造成很大的影响。

智慧:

*   永远不要将选取的对象存储在缓存中
*   总是存储压缩的 json blobs。

[https://gist.github.com/024c2a044bd802472b5b4bb19b3a279e](https://gist.github.com/024c2a044bd802472b5b4bb19b3a279e)

*   如果您想要存储它们，请确保在 Py2 和 Py3 上使用相同的协议。这个库有助于解决这个问题。我不建议这样做。例如，如果您存储了会话或令牌，这将对上游系统造成压力。

# 错误监视器

监控监控监控。如果你有大量的代码，没有单元测试会覆盖所有的极限情况。生产流量总是会有一些惊喜。在弹性 APM 或任何错误监控工具中记录所有异常。当您看到错误时，检查它们是否都在 Py2 和 Py3 中，或者它们是否是带有那些异常和调用堆栈的 Py3 特定日志票据。如果错误增加，回滚代码(关闭金丝雀流量)。提醒你，你总是有惊喜。这很好地展示了如何限制流量。

# 利益

迁移带给我们许多惊喜和意想不到的好处。

*   Python3.8 的内存消耗少了 20%多一点
*   这使我们使用的容器减少了 21%。我们运行 95–100 个任务，现在减少到 75 个。
*   网络吞吐量增加。
*   CPU 使用率没有太大变化
*   我们将平均响应时间从 110–120 秒降低到 90–100 秒。P95 和 P99 我们将它降低了近 100 毫秒——300 毫秒。

# 更少…..

*   假设您正在迁移您的主要应用程序和一些支持主要应用程序的内部库。在整个代码投入生产之前，请不要将代码提升到各自内部库的`master`分支。请在您的`feature`分支之外创建预发布([https://packaging . python . org/guides/distributing-packages-using-setup tools/# id68](https://packaging.python.org/guides/distributing-packages-using-setuptools/#id68))版本的内部库，并在主代码中使用它们，直到一切稳定。当你因为任何原因不得不恢复时，这将省去很多麻烦。

# 技术工具和微小差异

1.  `modernize`是比别人更好的选择

[https://gist.github.com/2d9b941fab741135ead1b57b1f494d78](https://gist.github.com/2d9b941fab741135ead1b57b1f494d78)

1.  解码和编码— —检查你在哪里使用了字节串——小心
2.  请注意`enum`它们在 Python 3 中略有不同
3.  当心`copy`功能
4.  `cmp`Python 3 中没有吗
5.  用 lambda 替换`string.lower`
6.  如果你在测试用例中使用`gevent`来猴子补丁，使用这个 Anthony 包- `pytest-gevent`
7.  用`caniusepython3`了解哪些库不适合。

# 一些常见的错误

1.  RuntimeError:字典在迭代过程中改变了大小
2.  类型错误:'<=’ not supported between instances of ‘str’ and ‘int’

[https://gist.github.com/272edf113ad8130c0b7c1755da504957](https://gist.github.com/272edf113ad8130c0b7c1755da504957)

# 参考

1.  [https://www . techrepublic . com/article/migrating-from-python-2-to-python-3-a-guide-to-preparating-for-the-2020-deadline/](https://www.techrepublic.com/article/migrating-from-python-2-to-python-3-a-guide-to-preparing-for-the-2020-deadline/)
2.  [https://medium . com/@ boxed/moving-a-large-old-code base-to-python 3-33 a5a 13 F8 c 99](/@boxed/moving-a-large-and-old-codebase-to-python3-33a5a13f8c99)
3.  [http://python3porting.com/preparing.html](http://python3porting.com/preparing.html)
4.  [https://www . digital ocean . com/community/tutorials/how-to-port-python-2-code-to-python-3](https://www.digitalocean.com/community/tutorials/how-to-port-python-2-code-to-python-3)
5.  [https://dive into python 3 . net/case-study-porting-char det-to-python-3 . html](https://diveintopython3.net/case-study-porting-chardet-to-python-3.html)
6.  [https://link . springer . com/content/pdf/bbm % 3a 978-1-4302-2416-7% 2 f1 . pdf](https://link.springer.com/content/pdf/bbm%3A978-1-4302-2416-7%2F1.pdf)
7.  [https://docs . python . org/3/reference/data model . html # object . bool](https://docs.python.org/3/reference/datamodel.html#object.__bool__)
8.  [https://stack overflow . com/questions/55695479/type error-not-supported-between-instances-of-dict-and-dict](https://stackoverflow.com/questions/55695479/typeerror-not-supported-between-instances-of-dict-and-dict)