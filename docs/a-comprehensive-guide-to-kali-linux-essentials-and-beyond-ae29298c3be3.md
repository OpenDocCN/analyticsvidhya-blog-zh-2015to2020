# Kali Linux 综合指南:要点及其他

> 原文：<https://medium.com/analytics-vidhya/a-comprehensive-guide-to-kali-linux-essentials-and-beyond-ae29298c3be3?source=collection_archive---------3----------------------->

![](img/e9a14939702742b6144eafff9cf75d09.png)

照片由[作者](https://www.instagram.com/dj_techguy/)拍摄

> 我认为从大的方面来看，Linux 是一件伟大的事情。这是一个伟大的黑客工具，它有很大的潜力成为更多的东西。~杰米·扎温斯基(jwz)

# 什么是 KALI LINUX？

Kali Linux 是基于 Debian 的 Linux 发行版，旨在进行高级渗透测试和安全审计。Kali Linux 包含数百个工具，这些工具适用于各种信息安全任务，例如渗透测试、安全研究、计算机取证和逆向工程。Kali Linux 由一家领先的信息安全培训公司进攻性安全公司开发、资助和维护。

马体·阿哈罗尼和迪冯·科恩斯是 Kali Linux 的核心开发人员。他们之前称之为回溯，宣传 Kali 是一个更好的继任者，有更多以测试为中心的工具，不像回溯有多个工具服务于相同的目的，反过来，使它挤满了不必要的实用程序。这使得使用 Kali Linux 的道德黑客和网络安全成为一项简化的任务。

要下载并安装最新版本的 Kali Linux 2020.4，请点击这里的官方指南[。](https://www.kali.org/docs/introduction/download-official-kali-linux-images/)

# 为什么你应该使用卡莉？

Kali Linux 是专门为渗透测试专业人员和安全专家的需求定制的，鉴于其独特的性质，如果您不熟悉 Linux 或者正在寻找用于开发、web 设计、游戏等的通用 Linux 桌面发行版，那么它不是推荐的发行版。

然而，如果你是一名专业渗透测试人员，或者正在以成为认证专家为目标学习渗透测试，没有比 Kali Linux 更好的工具包了，无论价格如何，因为——

1.  包括超过 **600 个渗透测试工具**
2.  **免费**使用*(如果你的时间没有价值)*
3.  基于**开源**开发模式运行
4.  给予**完全定制**的自由
5.  在**安全**环境中开发
6.  **自定义内核**，打补丁注入
7.  **ARMEL 和 ARMHF** 支架
8.  多样且充满活力的**社区**

# **卡莉的流行工具**

为了便于理解，我把 Kali Linux 最常用的/预装的工具分成不同的类别，并解释了它们的功能。

> **信息收集** —这些工具用于收集数据，并以一种可进一步使用的形式对数据进行格式化。

1.  **Nmap**

![](img/26584d148ca7b19b4a9784200ae74172.png)

Nmap 是世界上最著名的网络映射工具。它允许您发现任何网络中的活动主机，并获取与渗透测试相关的其他信息(如开放端口)。由于其易用性和强大的搜索扫描能力，它在黑客社区中获得了巨大的人气。

2.**网猫**

![](img/b2d76c3285586e98ca26f648e3cb28ec.png)

[Netcat](http://netcat.sourceforge.net/) 是一款网络探索应用，不仅在安全行业广受欢迎，在网络和系统管理领域也很受欢迎。虽然它主要用于出站/入站网络检查和端口探索，但当与 Perl 或 C 等编程语言或 bash 脚本结合使用时，它也很有价值。

3.**马尔特戈**

![](img/c588642b3e5f15d9830dc8d975f50cc4.png)

[Maltego](https://www.maltego.com/?utm_source=paterva.com&utm_medium=referral&utm_campaign=301) 是一款令人印象深刻的数据挖掘工具，用于在线分析信息并将信息联系起来。根据这些信息，它会创建一个有向图来帮助分析这些数据之间的联系。这将节省您的时间，并让您更准确、更智能地工作。Maltego 为您提供了一个更强大的搜索，给您更智能的结果。如果获取“隐藏”信息决定了你的成功，这个工具可以帮助你发现它。

> **漏洞分析** —这些工具用于检查系统或机器中存在的任何类型的流和漏洞，这些流和漏洞可能导致任何安全漏洞和数据丢失。

4.**床**

![](img/690428d17147eae36f2cd7f71992c22b.png)

[BED](https://gitlab.com/kalilinux/packages/bed) 代表暴力利用探测器。BED 是一个设计用来检查潜在缓冲区溢出、格式字符串错误等守护程序的程序。艾尔。这个工具只是向服务器发送命令，然后检查它是否还活着。当然，这不会检测指定守护进程的所有错误，但它将(至少应该)帮助您检查您的软件的常见漏洞。

5.**电源模糊器**

Powerfuzzer 是一个高度自动化和完全可定制的 web fuzzer(基于 HTTP 协议的应用程序 fuzzer ),基于许多其他可用的开源 fuzzer 和从众多安全资源和网站收集的信息。它被设计成用户友好、现代、有效和实用的。目前，它能够识别这些问题:跨站点脚本(XSS)、注入(SQL、LDAP、代码、命令和 XPATH)、CRLF 和 HTTP 500 状态(通常表示可能的错误配置/安全缺陷，包括缓冲区溢出)

> **Web 应用分析** —这些工具通过浏览器识别并访问网站，以检查任何可能导致任何信息或数据丢失的缺陷或漏洞。

6.**打嗝组曲**

![](img/58c5f2fea1b2cdf0f7ba19323e60ee9b.png)

Burp Suite 是最流行的 web 应用安全测试软件之一。它的各种工具无缝协作，支持整个测试过程，从应用攻击面的初始映射和分析，到发现和利用安全漏洞。Kali Linux 附带了 burp suite community edition，它是免费的，但这个工具有一个付费版本，称为 burp suite professional，与 burp suite community edition 相比，它有很多功能。

7. **OWASP ZAP**

![](img/e1ea2963547afb5e197d682766b7b998.png)

[OWASP Zed 攻击代理(ZAP)](https://www.zaproxy.org/) 是一个易于使用的集成渗透测试工具，用于查找 web 应用程序中的漏洞。它是为具有广泛安全经验的人设计的，因此非常适合初涉渗透测试的开发人员和功能测试人员，也是有经验的 pen 测试人员工具箱的有用补充。

8. **WPScan**

![](img/51783ad81b42ddff535b961656f59783.png)

WPScan 是一个黑盒 WordPress 漏洞扫描器，可以用来扫描远程 WordPress 安装以发现安全问题。通过使用 WPScan，你可以检查你的 WordPress 设置是否容易受到某些类型的攻击，或者它是否在你的核心、插件或主题文件中暴露了太多的信息。这个 WordPress 安全工具还可以让你找到所有注册用户的弱密码，甚至可以对它进行暴力攻击，看看哪些可以被破解。

9.**鲣鱼**

[Skipfish](http://code.google.com/p/skipfish/) 是一款主动的 web 应用安全侦察工具。它通过执行递归爬行和基于字典的探测为目标站点准备交互式站点地图。然后用大量主动(但希望是非破坏性的)安全检查的输出对生成的地图进行注释。该工具生成的最终报告旨在作为专业 web 应用程序安全评估的基础。

> **数据库评估** —这些工具用于访问数据库，并针对不同的攻击和安全问题进行分析

10. **sqlmap**

![](img/8ca19f7ff6fcfc428c7fa26329147f7f.png)

sqlmap 是一个开源渗透测试工具，可以自动检测和利用 SQL 注入漏洞，接管数据库服务器。它配备了一个强大的检测引擎，许多针对终极渗透测试器的利基功能，以及一系列广泛的开关，从数据库指纹识别，从数据库获取数据到通过带外连接访问底层文件系统和在操作系统上执行命令。

11. **BBQSQL**

[BBQSQL](https://github.com/Neohapsis/bbqsql/) 是一个用 Python 写的盲 SQL 注入框架。这在攻击棘手的 SQL 注入漏洞时非常有用。BBQSQL 也是一个半自动工具，允许对那些难以触发 SQL 注入发现的内容进行相当多的定制。该工具与数据库无关，并且非常通用。它也有一个直观的用户界面，使设置攻击更加容易。还实现了 Python gevent，使得 BBQSQL 速度极快。

> **密码攻击** —这些工具可用于通过不同的服务和协议检查任何登录凭证上的单词列表或密码列表。

12.**开膛手约翰**

![](img/5e2dc8f3c0f8a9e2925c9124b7ae1519.png)

开膛手约翰是有史以来最受欢迎的密码破解者之一。它还是测试操作系统密码强度或远程审计密码的最佳安全工具之一。这个密码破解程序能够自动检测几乎任何密码中使用的加密类型，并相应地改变其密码测试算法，使其成为有史以来最智能的密码破解工具之一。

13. **THC-Hydra**

![](img/ed197e1489872f3264725868e2c53fa9.png)

[Hydra](https://gitlab.com/kalilinux/packages/hydra) 是一个支持多种协议攻击的并行登录破解程序。它非常快速和灵活，新的模块很容易添加。该工具使研究人员和安全顾问能够展示远程获得对系统的未授权访问是多么容易。如果你正在寻找一个有趣的工具来破解登录/密码对，Hydra 将是预装的最好的 Kali Linux 工具之一。

14.**嘎吱声**

[Crunch](https://sourceforge.net/projects/crunch-wordlist/) 是一个单词列表生成器，在这里你可以指定一个标准字符集或者你指定的字符集。crunch 可以产生所有可能的组合和排列。Crunch 可以以组合和排列的方式生成单词表，它还可以根据行数或文件大小来分解输出。

> **无线攻击** —这些工具是无线安全黑客，像破坏 wifi 路由器、工作和操纵接入点。

15.**空气破裂**

![](img/967df21f15b015837fedffc29915f315.png)

[Aircrack-ng](https://www.aircrack-ng.org/) 是一款无线安全软件套件。它包括一个网络数据包分析器、一个 WEP 网络破解程序、WPA / WPA2-PSK 以及另一套无线审计工具。它有助于捕获包并从中读取哈希，甚至通过各种攻击(如字典攻击)来破解这些哈希。

16.**蕨类 Wifi 破解**

![](img/79c7e3e3ce5dcbb029e93e7a07c34db9.png)

[Fern Wifi Cracker](https://github.com/savio-code/fern-wifi-cracker) 是一个使用 Python 编程语言和 Python Qt GUI 库编写的无线安全审计和攻击软件程序，该程序能够破解和恢复 WEP/WPA/WPS 密钥，还可以在无线或以太网上运行其他基于网络的攻击。

17.**命中注定**

[Kismet Wireless](https://www.kismetwireless.net/) 是一款多平台免费无线局域网分析器、嗅探器、IDS(入侵检测系统)。它几乎兼容任何种类的无线网卡。在嗅探模式下使用它，您可以使用 802.11a、802.11b、802.11g 和 802.11n 等无线网络。它可以使用其他程序来播放网络事件的音频警报、读取网络摘要或提供 GPS 坐标。

> **逆向工程** —这些工具可以用来分解应用程序或软件的各个层，以获得应用程序的源代码，了解其工作原理，并根据需要进行操作。

18. **Apktool**

![](img/d7470dab099cd66e2135222d40b0c37a.png)

Apktool 的确是 Kali Linux 上用于逆向工程 Android 应用的流行工具之一。它可以将资源解码为几乎原始的形式，并在进行一些修改后重建它们；使得一步一步调试 smali 代码成为可能。此外，由于类似项目的文件结构和一些重复任务(如构建 apk 等)的自动化，它使使用应用程序变得更加容易。

19.**奥利德格**

[OllyDbg](http://www.ollydbg.de/) 是一款用于微软 Windows 的 32 位汇编级分析调试器。对二进制代码分析的强调使得它在源代码不可用的情况下特别有用。它跟踪寄存器、识别过程、API 调用、开关、表、常数和字符串，并从目标文件和库中定位例程。它有一个用户友好的界面，其功能可以通过第三方插件进行扩展。

> **开发工具** —这些工具用于开发不同的系统，如个人电脑和手机。他们可以为易受攻击的系统生成有效负载，通过这些有效负载，可以利用设备中的信息。

20. **Metasploit 框架**

![](img/d856532082dcea79e9b81372eab09d6d.png)

[Metasploit](https://metasploit.com/) 是一个基于 Ruby 的平台，用于开发、测试和执行针对远程主机的攻击。它包括一整套用于渗透测试的安全工具，以及一个强大的基于终端的控制台，称为 msfconsole，允许您找到目标，启动扫描，利用安全漏洞并收集所有可用的数据。它还允许您复制网站，用于网络钓鱼和其他社会工程目的。Metasploit 是一个 CLI 工具，但它甚至有一个名为 [Armitage](https://tools.kali.org/exploitation-tools/armitage) 的 GUI 包，这使得 Metasploit 的使用更加方便和可行。

21. **SearchSploit**

![](img/04ecc7e0d551f374a00afe97f22ff917.png)

SearchSploit 是 Exploit-DB 的命令行搜索工具，它还允许您随身携带漏洞数据库的副本。SearchSploit 使您能够通过本地签出的存储库副本执行详细的离线搜索。这种能力对于在没有互联网接入的隔离或空气间隙网络上的安全评估特别有用。

22.**牛肉**

[BeEF](https://beefproject.com/) 代表浏览器开发框架。它是一个专注于网络浏览器的渗透测试工具。BeEF 允许专业渗透测试人员使用客户端攻击媒介来评估目标环境的实际安全状况。

> **嗅探和欺骗** —这些工具用于秘密访问/窃听网络上任何未经授权的数据，或者隐藏真实身份并创建假身份以用于任何非法或未经授权的工作。

23. **Wireshark**

![](img/fb72fb51c358f4f4e6ff8bb8118b8aa9.png)

Wireshark 是一款免费的开源软件，可以让你实时分析网络流量。由于其嗅探技术，Wireshark 因其检测任何网络中的安全问题的能力以及解决一般网络问题的有效性而广为人知。在嗅探网络时，您能够截取并读取人类可读格式的结果，这使得识别潜在问题(例如低延迟)、威胁和漏洞变得更加容易。

24. **Bettercap**

![](img/b48008009ece24b9be96323a30c1592d.png)

[Bettercap](https://www.bettercap.org/) 是网络攻击和监控的瑞士军刀。它是一个网络安全工具，用于网络捕获、分析和 MITM 攻击。Bettercap 最直接的用途是使用扫描和侦察模块来识别附近的目标，然后在捕获必要的信息后，尝试识别具有弱密码的网络。

> **后利用** —这些工具使用后门回到易受攻击的系统，即保持对机器的访问。

25. **Powersploit**

![](img/33837cf7140a4b0476608eeda7dc6e0b.png)

[PowerSploit](https://github.com/PowerShellMafia/PowerSploit) 是一个开源的攻击性安全框架，由 PowerShell 模块和脚本组成，执行与渗透测试相关的广泛任务，如代码执行、持久性、绕过防病毒、recon 和渗透。

26.**甜蜜地**

![](img/cfe498f689ef2f06f2038c8d5334deae.png)

Weevely 是一个隐藏的 PHP web shell，模拟类似 telnet 的连接。它是 web 应用程序后期开发的重要工具，可以用作秘密后门或 web 外壳来管理合法的 web 帐户，甚至是免费托管的帐户。

27.**http 隧道**

[HTTPTunnel](https://www.gnu.org/software/httptunnel/) 是一个隧道软件，它可以通过纯粹的 HTTP“GET”和“POST”请求上的限制性 HTTP 代理来隧道化网络连接。HTTPTunnel 创建在 HTTP 请求中通过隧道传输的双向虚拟数据流。如果需要，可以通过 HTTP 代理发送请求。这对限制性防火墙后的用户非常有用。

> **取证** —取证专家使用这些工具从任何系统或存储设备中恢复信息。

28.**尸检**

![](img/e58e117b42816d592cac7c45a0d0cc15.png)

[验尸](https://www.autopsy.com/)是一个快速数据恢复和哈希过滤的法医实用程序。此工具使用 PhotoRec 从未分配的空间中切割已删除的文件和媒体。它还可以提取 EXIF 扩展多媒体。尸检扫描妥协指标使用 STIX 图书馆。它在命令行和 GUI 界面中都可用。

29.**宾沃克**

![](img/0818205b1d9f32756ccdaa88c7d3395e.png)

[Binwalk](https://github.com/ReFirmLabs/binwalk) 是一个在给定的二进制映像中搜索嵌入文件和可执行代码的工具。具体来说，它是为识别固件映像中嵌入的文件和代码而设计的。Binwalk 使用 libmagic 库，因此它与为 Unix 文件实用程序创建的 magic 签名兼容。它还包括一个定制的魔术签名文件，该文件包含固件映像中发现的文件的改进签名，如压缩/归档文件、固件头文件、Linux 内核、引导加载程序、文件系统等。

30.**批量 _ 提取器**

[bulk_extractor](https://github.com/simsong/bulk_extractor/) 是一个从数字证据文件中提取电子邮件地址、信用卡号、URL 和其他类型信息的程序。它是一个有用的取证调查工具，用于许多任务，如恶意软件和入侵调查、身份调查和网络调查，以及分析图像和密码破解。

31. **pdf 解析器**

[pdf-parser](https://blog.didierstevens.com/programs/pdf-tools/) 是 pdf 文件最重要的取证工具之一。pdf-parser 解析一个 pdf 文档，并区分在分析过程中使用的重要元素，而该工具不呈现该 pdf 文档。

> **报告工具** —这些工具开发统计数据和信息，以帮助在所有评估和漏洞测试完成后，以一种有组织的和经过验证的方式进行分析和报告。

32.**德拉迪斯**

![](img/cc04c2523af62a266405f960a8d72c5d.png)

Dradis 是一个开源框架，能够实现有效的信息共享，尤其是在安全评估期间。Dradis 是一个自包含的 web 应用程序，它提供了一个集中的信息库来跟踪到目前为止已经完成的工作，以及未来的工作。简单的报告生成、对附件的支持、通过服务器插件与现有系统和工具的集成是它的一些显著特点。

33.**魔法树**

MagicTree 是一款渗透测试生产力工具。它旨在允许简单直接的数据整合、查询、外部命令执行和(耶！)报告生成。如果你想知道，“树”是因为所有的数据都存储在一个树结构中，“神奇”是因为它被设计成神奇地完成渗透测试中最繁琐和无聊的部分——数据管理和报告。

34. **Metagoofil**

[metagoufil](http://www.edge-security.com/metagoofil.php)是一个信息收集工具，用于提取属于目标公司的公共文档(pdf、doc、xls、ppt、docx、pptx、xlsx)的元数据。Metagoofil 将在 Google 中执行搜索以识别文档并将其下载到本地磁盘，然后将使用不同的库(如 Hachoir、PdfMiner？和其他人。根据结果，它生成一个包含用户名、软件版本和服务器或机器名称的报告，这将对渗透测试人员有所帮助。

> **社会工程** —这些工具生成人们在日常生活中使用的类似服务，并使用这些虚假服务提取个人信息。这些工具使用和操纵人类行为来收集信息。

35.**设置**

![](img/6809c150e405a26db2ebc60db5529f21.png)

[社会工程工具包](https://github.com/trustedsec/social-engineer-toolkit)(称为 SET)是一个开源的基于 Python 的渗透测试框架，专为社会工程而设计。SET 有许多自定义的攻击向量，允许你在很短的时间内进行可信的攻击。

36. **U3-Pwn**

[U3-Pwn](http://www.nullsecurity.net/tools/backdoor.html) 是一款工具，旨在通过默认 U3 软件安装将可执行文件自动注入 Sandisk 智能 USB 设备。这是通过从设备中删除原始 iso 文件并创建一个具有自动运行功能的新 iso 来实现的。

37.**wifi fisher**

[wifi fisher](https://wifiphisher.org/)是一种安全工具，它对 Wi-Fi 网络发起自动网络钓鱼攻击，以获取凭据或使受害者感染恶意软件。这是一种社会工程攻击，可用于获取 WPA/WPA2 密码，与其他方法不同，它不需要任何暴力。在使用 Evil Twin 攻击实现中间人位置后，Wifiphisher 将所有 HTTP 请求重定向到攻击者控制的钓鱼页面。

# 学习 KALI LINUX

![](img/50dd5dd535e3a944bb63c0ab58fcbb0d.png)

部署在 [VirtualBox](https://www.virtualbox.org/) 上的 Kali-Linux 虚拟机的快照

简单来说，学习一件事最好的方法就是去体验。也就是说，每天使用卡利。从建立一个 **Kali Linux 虚拟机**开始，接下来学习它们提供的工具和所有功能，并尽可能多地练习。设置一个易受攻击的虚拟机，通过将其用作目标来进一步练习，或者尝试侵入您有权攻击的网站，幸运的是，有几个在线网站正是为此目的而设置的。互联网上有大量的教程、演练和文档，好好利用这些资源，你很快就会掌握卡利语。

想入门 Kali Linux，但不确定如何入门？查看**进攻安全** — [卡莉 Linux 揭示书](https://kali.training/)的官方指南，或者你也可以在 [Udemy](https://www.udemy.com/topic/kali-linux/) 上查看一些在线付费课程，或者从 YouTube 上的大量免费教程视频中学习，比如[这个](https://www.youtube.com/watch?v=lZAoFs75_cs)。

# 结论

请记住，Kali Linux 虽然不太复杂，但并不完全适合初学者，所以在使用这些工具时要慢慢来。如果你是 Linux 世界的新手，可以考虑从另一个 Linux 系统开始，比如 Ubuntu，体验一下你将会接触到的东西。记住你需要的所有工具都是免费的。从免费的虚拟机管理程序到网络安全工具和 Kali Linux 本身，学习它主要是对你的时间和精力的投资。

唷！那真是一篇很长的文章。希望您能够了解到世界上最好的道德黑客和渗透测试套件的本质，并且现在对 Kali Linux 提供的大量网络安全工具有了更好的总体理解。

> 支持我 https://www.buymeacoffee.com/djrobin17

# 参考

[](https://www.kali.org/docs/) [## Kali Linux -文档

### Kali Linux 的官方文档，一个用于渗透测试的高级渗透测试 Linux 发行版…

www.kali.org](https://www.kali.org/docs/) [](https://www.tutorialspoint.com/kali_linux/index.htm) [## Kali Linux 教程

### Kali Linux 是一个有道德的黑客的最好的开源安全包之一，包含一组工具，由…

www.tutorialspoint.com](https://www.tutorialspoint.com/kali_linux/index.htm) [](https://www.geeksforgeeks.org/kali-linux-tools/?ref=lbp) [## Kali Linux 工具- GeeksforGeeks

### Kali Linux 是一个基于 Linux 的操作系统，主要用于渗透测试。Kali.org 最近发布了它的…

www.geeksforgeeks.org](https://www.geeksforgeeks.org/kali-linux-tools/?ref=lbp) [](https://securitytrails.com/blog/kali-linux-penetration-testing-tools) [## security trails | 25 大 Kali Linux 渗透测试工具

### Kali 最大的优点之一是它不需要你在硬盘上安装操作系统——它…

securitytrails.com](https://securitytrails.com/blog/kali-linux-penetration-testing-tools)  [## 关于 Kali Linux-StartaCyberCareer.com 你需要知道的 7 件事

### 这篇文章是关于 Kali Linux 发行版以及网络安全专家如何使用它的。如果你想找到…

startacybercareer.com](https://startacybercareer.com/7-things-you-need-to-know-about-kali-linux/) 

***嘿！如果你碰巧滚动到这里来查看这篇文章有多长，我只想告诉你，你是牛逼的:D***