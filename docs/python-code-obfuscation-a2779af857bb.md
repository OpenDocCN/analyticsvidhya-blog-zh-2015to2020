# Python 代码混淆

> 原文：<https://medium.com/analytics-vidhya/python-code-obfuscation-a2779af857bb?source=collection_archive---------10----------------------->

![](img/89a9d18f6b0c9060ab7e27dd3639d4ef.png)

照片由[乔希·布特](https://unsplash.com/@joshboot?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

您正在使用 Python 并直接在 docker 镜像中或通过 [pip](https://pip.pypa.io/en/stable/) 可安装包部署您的代码。没有额外的预防措施，你的源代码很容易被任何人访问。如果您不介意的话，您可能仍然希望防止任何人篡改代码。由于 Python 的解释性质，这样做需要额外的步骤。谢天谢地，像往常一样，在丰富的 Python 生态系统中:解决方案是存在的。下面列出了针对此问题的现有解决方案的类别以及所选解决方案的详细信息。

*免责声明—* 有人告诉我，只要付出足够的努力，任何事情都可以被逆向工程。我们的目标是让这变得更加困难。此外，基于**信任**做出固执己见的选择。这一次**闭源**解决方案是首选，因为恶意用户没有机会查看它的实现细节。

*   **Python to exe family**
    将 Python 代码作为可执行文件的工具将解释器和源代码捆绑到一个二进制文件中。无需额外的预防措施，即可轻松访问该资源。这些工具应该与下面描述的其他技术一起使用！
*   **混淆** 简单的 python 混淆例如 base64 编码很容易被破解。很容易猜到…
*   **通过加密进行混淆**
    这些更先进，允许额外的功能，如在加密代码的基础上设置过期时间(TTL)。(强制过期时间:未测试)
*   **一些更复杂的技术(比如 dropbox)也以逆向工程告终。关于这方面的有趣阅读，请点击这些[链接](https://reverseengineering.stackexchange.com/questions/22648/best-way-to-protect-source-code-of-exe-program-running-on-python?)。**
*   cyt honcyt hon
    特别值得一提的是 Cython ，它值得拥有自己的类别。使用 Cython，您可以将 Python 代码编译成 C 二进制文件，这也使得逆向工程变得足够困难。然而，选择这种方式可能会涉及更改代码和额外的工作。

# 用 sourcedefender 加密 python 源代码

选择的解决方案是: [SourceDefender](https://pypi.org/project/sourcedefender/) ，因为它很简单:

*   您只需要运行一个命令来加密您想要保护免受**攻击的源代码片段。py** 至**。pye** 。
*   也因为他们通过快速解决一个 bug 而提供的技术支持。

> SOURCEdefender 可以用 AES 256 位加密保护你的明文 Python 源代码。由于解密过程发生在模块导入期间或在命令行上加载脚本时，因此不会影响正在运行的应用程序的性能。一旦从*加载，加密代码的运行速度不会变慢。与从*加载相比，pye* 文件。py* 或*。pyc* 文件。

从 pypi 页面可以看出，基本用法很简单。一旦运行受保护的代码，像 Python 的 inspect 模块的 [getsource](https://docs.python.org/3/library/inspect.html#inspect.getsource) 这样的基本窥探工具就会返回乱码。在错误回溯中，包括文件名和行号，但也没有源。

```
Traceback (most recent call last):
  File "src/config/helper.pye", line 85, in start
IxlqeEBtjE9CDDaHwLr395P6GeE1hjM07faWWV+D6ytu7lgbRXBeHGlRt...
  File "src/config/settings.pye", line 49, in get_service
kIAsrHxOSEXRBwKAPMe7Ys9GY85aT5J+d9muxXUoMFeajjZFCsvPwG121...
ImportError: cannot import name '+++++'
```

当通过在 docker 映像中复制代码来部署代码时，这就是您所需要的。如果您部署为 python 包，请继续阅读。

# Python 包加密

为了用加密的 python 代码创建 Pip 安装包，你需要欺骗 [setuptools](https://setuptools.readthedocs.io/en/latest/) ，并让它打包 ***。pye** 文件给你。历史上，distutils 和 setuptools 依赖于`__init__.py`文件的存在。使用 Python3 的[隐式名称空间包](https://www.python.org/dev/peps/pep-0420/)，允许创建没有`__init__.py`文件的包，以及 setuptools 中的相应功能:[find _ Namespace _ Packages](https://setuptools.readthedocs.io/en/latest/userguide/package_discovery.html#using-find-namespace-or-find-namespace-packages)，您需要在您的`setup.py`文件中做的就是:

*   用`find_namespace_packages` 代替`find_packages`
*   捆绑所有 ***。pye** 文件为`package_data`

# 完整 setup.py 示例

```
try:
    import sourcedefender
except ModuleNotFoundError:
    passimport osfrom setuptools import setup, find_namespace_packagesPROJECT = "hello_world"def package_pye_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.pye'):
                paths.append(os.path.join('..', path, filename))
    return pathspye_files = package_pye_files(f"./{PROJECT}")PACKAGE_DATA = {
    PROJECT: ["./resources/*"] + pye_files
}with open("requirements.txt", "r") as f:
    REQUIREMENTS = f.read().splitlines()setup(
    name=PROJECT,
    version="0.0.1",
    description="hello_world",
    author="None",
    python_requires='>=3',
    packages=find_namespace_packages(include=[f"{PROJECT}*"]),
    package_data=PACKAGE_DATA,
    include_package_data=True,
    install_requires=REQUIREMENTS
)
```

# 参考

*   [python 包分发类型](/ochrona/understanding-python-package-distribution-types-25d53308a9a)
*   [python 应用混淆和许可](https://vigneshgig.medium.com/python-application-obfuscate-and-licensing-7531f06d6296)