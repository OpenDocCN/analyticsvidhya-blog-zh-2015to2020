# 使用 CMake 的 FetchContent 进行 C++依赖管理

> 原文：<https://medium.com/analytics-vidhya/c-dependency-management-with-cmakes-fetchcontent-4ceca4693a5d?source=collection_archive---------7----------------------->

![](img/42670a1309758a13bc9e10ffb4529908.png)

照片由[埃文·克劳斯](https://unsplash.com/@evankrause_?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在这篇文章中，我们将讨论 C++中的依赖管理。特别介绍一下`FetchContent`，一个应该会得到更多喜爱的 CMake 特性！

我们从 C++应用程序的依赖管理概述开始。

# Git 子模块

过去，我的默认做法是将每个依赖项作为 git 子模块添加。对于现代 CMake 设置的依赖项，这足以使它可用于您的项目。

但是我一直不喜欢这种方法，因为它要求每个用户都运行

```
$ git submodule update --init
```

在构建代码之前。我喜欢让人们使用我的代码变得简单。因此，我一直在寻找一种方法来摆脱运行任何额外的命令。此外，我已经多次发现自己忘记运行`git submodule update --init`。

# 将代码复制到您的存储库中

当然，您也可以将依赖项的整个源代码复制到项目的存储库中。然而，在我看来，这不如子模块。更新依赖关系变得复杂。而且，根据您的依赖关系有多大，这会使您的存储库膨胀。

# 柯南(和其他包装经理)

另一个选择是使用像[柯南](https://conan.io/)这样的包管理器。我发现它们是更大项目的合适解决方案。当你有需要很长时间构建的依赖项时，它们也很好(我在看你的 OpenCV！😴).
但是对于需要 Github 的两到三个库的小型项目来说，这往往太费力了。

我的主要批评点是包管理器本身变成了一个依赖项。每个想要构建您的项目的人都已经安装了正确的版本。如果您正在使用 CI，您还必须在构建服务器上设置所有的东西，这有时会很麻烦。

# CMake 能帮我们吗？

Rust 和 Go 等较年轻的语言在构建系统中加入了包管理。这有助于更好的开发者体验，因为用户不必选择使用哪个包管理器。此外，默认情况下，软件包是为内置软件包管理器的语言设置的。

考虑到这一点，很自然地会向 CMake 寻求解决方案。听说 CMake 在 3.0 版本中引入了一个名为`[ExternalProject](https://cmake.org/cmake/help/latest/module/ExternalProject.html)`的模块，我很兴奋。`ExternalProject`将依赖项包装到 CMake 目标中，并允许从您的`CMakeLists.txt`管理外来代码。

要使用它，必须通过`ExternalProject_Add()`添加一个目标。然后，CMake 将为此目标运行以下步骤。

`DOWNLOAD`

下载依赖项。这里可以使用版本控制系统或从 URL 下载。

`UPDATE`

如果自上次 CMake 运行以来发生了任何变化，请更新下载的代码。

`CONFIGURE`

配置项目代码。

`BUILD`

构建依赖项代码。

`INSTALL`

将构建的代码安装到指定的目录中。

`TEST`(可选)

运行测试。

以上所有命令都是可配置的。`ExternalProject`也允许自定义步骤。有关更多信息，请查看[文档](https://cmake.org/cmake/help/latest/module/ExternalProject.html#module:ExternalProject)。

# 那么，这就是解决办法吗？

稍微摆弄了一下，发现`**ExternalProject**` **不是我要找的。**😕

原因是当使用`ExternalProject`时，它的所有步骤都将在构建时运行。这意味着 CMake 会在生成步骤之后下载并构建您的依赖项。因此，当 CMake 配置您的项目时，您的依赖项还不可用。

# 那么我们必须坚持子模块吗？

不，我们不会！

在 3.11 版本中，CMake 引入了一个新的模块:`[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)`。该模块提供与`ExternalProject`相同的功能，但将在配置时间之前下载依赖关系。这意味着我们可以用它来管理来自`CMakeLists.txt`文件的 C++项目依赖关系！🎉🎉🎉

# 如何使用`FetchContent`

在这个[资源库](https://github.com/bewagner/fetchContent_example)中，我准备了一个例子。它使用`FetchContent`来包含库的 [doctest](https://github.com/onqtam/doctest) 和 [range-v3](https://github.com/ericniebler/range-v3) 。CMake 将下载并构建依赖项。非常方便！我将在下面解释一切是如何工作的。确保你得到了[代码](https://github.com/bewagner/fetchContent_example)来使用它。

我们首先创建一个常规的 CMake 项目。在 3.14 版本中，`FetchContent` API 得到了改进，使得使用更加容易。因此，这是我们需要的最少版本。然后我们包括了`FetchContent`模块。

`cmake_minimum_required(VERSION 3.14)
project(fetchContent_example CXX)
include(FetchContent)`

我们通过调用`FetchContent_Declare()`来注册每个依赖项。进行此调用时，您可以自定义如何加载依赖项。`FetchContent`理解与`ExternalProject`几乎相同的选项。但是与`CONFIGURE`、`BUILD`、`INSTALL`和`TEST`相关的选项被禁用。

我们声明两个目标，一个用于`doctest`，一个用于`range-v3`。CMake 通过 git 库下载这两个库。

参数`GIT_TAG`指定了我们使用的依赖关系历史中的提交。这里也可以使用 git 分支名称或标记。然而，新的提交可以改变分支所指向的内容。这可能会影响项目的可重复性。所以 CMake 文档不鼓励使用分支名称或标记。

```
FetchContent_Declare(DocTest 
    GIT_REPOSITORY "https://github.com/onqtam/doctest"         
    GIT_TAG "932a2ca50666138256dae56fbb16db3b1cae133a" ) FetchContent_Declare(Range-v3         
    GIT_REPOSITORY "https://github.com/ericniebler/range-v3" 
    GIT_TAG "4d6a463bca51bc316f9b565edd94e82388206093" )
```

接下来，我们调用`FetchContent_MakeAvailable()`。这个调用确保 CMake 下载我们的依赖项并添加它们的目录。

`FetchContent_MakeAvailable(DocTest Range-v3)`

最后，我们可以添加一个可执行文件并链接到包含的包。CMake 接手了所有的重活！

```
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} doctest range-v3)
```

使用 git 存储库是包含依赖于`FetchContent`的最方便的方式。但是，如果您所依赖的代码不是 git 存储库，您可以定制`FetchContent`来使用其他来源。看看 `[ExternalProject](https://cmake.org/cmake/help/latest/module/ExternalProject.html#module:ExternalProject)`的[文档。它解释了所有参数。](https://cmake.org/cmake/help/latest/module/ExternalProject.html#module:ExternalProject)

整个`CMakeLists.txt`文件会是这样的。

```
cmake_minimum_required(VERSION 3.14) 
project(fetchContent_example CXX)  
include(FetchContent)  
FetchContent_Declare(DocTest 
    GIT_REPOSITORY "https://github.com/onqtam/doctest"         
    GIT_TAG "932a2ca50666138256dae56fbb16db3b1cae133a" ) FetchContent_Declare(Range-v3         
    GIT_REPOSITORY "https://github.com/ericniebler/range-v3"  
    GIT_TAG "4d6a463bca51bc316f9b565edd94e82388206093" ) FetchContent_MakeAvailable(DocTest Range-v3) add_executable(${PROJECT_NAME} src/main.cpp) target_link_libraries(${PROJECT_NAME} doctest range-v3)
```

在 CMake 中设置好一切之后，我们可以在源代码中使用这些包。下面是一个小的测试程序，它使用了两个包含的库。

```
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN  
#include <vector> 
#include "doctest/doctest.h" 
#include "range/v3/view.hpp" 
#include "range/v3/numeric/accumulate.hpp" 

int f(const std::vector<int> &v) {
    auto square = [](int i){ return i * i; };      
    int sum = ranges::accumulate(v
        | ranges::views::transform(square)       
        | ranges::views::take(10)
        , 0);

    return sum; 
    } TEST_CASE ("Test function") { 
CHECK(f({1, 2, 3}) == 14);             
CHECK(f({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) == 385);             CHECK(f({}) == 0); 
}
```

完整的示例项目是[这里的](https://github.com/bewagner/fetchContent_example)。

# 需要注意什么

以下是使用`FetchContent`时需要记住的一些事情。

## 下载需要互联网连接

当然，首先你必须在线下载你的依赖项。与使用子模块相比，这个需求现在是隐藏的。因此，在构建代码时，您可能会忘记它。为了缓解这个问题，有一组选项。

*   `FETCHCONTENT_FULLY_DISCONNECTED=ON`将跳过`DOWNLOAD`和`UPDATE`步骤
*   `FETCHCONTENT_UPDATES_DISCONNECTED=ON`将跳过`UPDATE`步骤

## 输出会变得非常冗长

`FetchContent`将记录其所有步骤。这就是控制台输出变得难以阅读的原因。要使所有输出静音，请将`FETCHCONTENT_QUIET`设置为`ON`。

## 该库必须是可安装的

我经常遇到的一个问题是，我想要使用的依赖项是不可安装的。每隔一段时间，您会遇到在它们的`CMakeLists.txt`中缺少对`install()`的调用的库。在这种情况下，`FetchContent`不知道如何将构建好的代码复制到安装文件夹中，将会失败。在这种情况下，考虑添加`install()`呼叫并创建一个 PR。

`FetchContent`最适合基于 CMake 的依赖项。我还没有机会用不是用 CMake 构建的库来测试它。但是我认为需要一些额外的配置来使它工作。

# 结论

在这篇博文中，我们了解了`FetchContent`。现在您知道了如何在 CMake 设置中管理您的依赖项。我们看到了如何使用`FetchContent`来获取一个小示例项目的依赖关系。我们还读到了一些使用时需要注意的事情。

我很喜欢这种管理依赖关系的方式。当然，你可以混合不同的方法，使用最适合你的方法！

如果你认为我在这篇文章中遗漏了什么，请告诉我！

如果你喜欢我的写作，考虑支持我，这样我可以继续为你创造内容！

[![](img/442e2379dd4c422211e3762adb3e50e2.png)](https://ko-fi.com/bewagner)[![](img/8f52df8c73b6eb1b2ef850f39042e120.png)](https://www.patreon.com/bewagner?fan_landing=true)

**在 Twitter 上关注我**[**@ be Wagner _**](https://twitter.com/bewagner_)**了解更多关于编程和 C++的想法！**

这篇文章最初发表在我的博客上:

[https://be Wagner . net/programming/2020/05/02/cmake-fetch content/](https://bewagner.github.io/programming/2020/05/02/cmake-fetchcontent/)