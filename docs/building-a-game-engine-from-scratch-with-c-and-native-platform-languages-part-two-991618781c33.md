# 用 C++和本地平台语言从头开始构建游戏引擎(第二部分)

> 原文：<https://medium.com/analytics-vidhya/building-a-game-engine-from-scratch-with-c-and-native-platform-languages-part-two-991618781c33?source=collection_archive---------2----------------------->

![](img/88abe32550422e8ce3d3672eb8156c84.png)

让我们开始吧！本系列将利用终端，所以您将需要复习一下如何使用操作系统的命令行。我会解释这是怎么回事，但这将取决于你为你的操作系统找出它。

因此，请在您最喜欢的代码编辑器中打开我们的项目。此外，弹出打开一个终端和 cd(更改目录)到项目的根目录。让我们编写我们创建的 CMakeLists.txt 文件的第一行。

第一行需要是我们希望这个文件使用的 CMake 的最低版本。这将向 CMake 发出信号，如果安装了足够新版本的 CMake，就只尝试构建我们的项目。

在终端中，输入命令…

```
cmake --version
```

输出将告诉您已经安装的 CMake 的版本。我的写着“cmake 版本 3.16.1”。复制这个数字，让我们把这一行添加到我们的文件的顶部。

```
cmake_minimum_required(VERSION 3.16.1 FATAL_ERROR)
```

太好了！我们的引擎已经正式启动了！干得好。下一步是命名我们的项目。将这些行添加到文件中。

```
project(engine)set(CMAKE_CXX_STANDARD 11)
```

现在我们已经告诉 CMake 我们项目的名称，以及我们将使用哪个版本的 C++。对我们来说，C++11 就是我们所需要的。

接下来，我们将添加一个静态库目标，这将是我们的引擎，以及一个可执行目标，这只是为了测试。稍后我们将创建本地平台项目来运行我们的游戏。但是现在，我们只是想要一个快速的方法来测试我们写的代码。

首先，让我们定义我们的源代码文件。我们还没有创建任何 C++文件，但接下来我们会这样做。

将这几行添加到 CMakeLists.txt 的底部

```
set(ENGINE_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/engine.h ${CMAKE_CURRENT_LIST_DIR}/src/engine.cpp)add_library(engine STATIC ${ENGINE_SOURCE}) set(TESTING_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/testing.cpp)
add_executable(program ${TESTING_SOURCE}) target_link_libraries(program engine)
```

因此，我们已经定义了一个静态库目标和一个可执行文件，并为每个目标分配了一些源代码文件。希望这是不言自明的。CMAKE_CURRENT_LIST_DIR 只是一个变量，它存储磁盘上 CMAKE 文件的绝对路径。

让我们继续添加这些源文件，并在其中填充一些代码！

让我们创造。/src/engine.h 优先。现在我们将保持它非常简单。

```
#pragma oncevoid Initialise();
```

…还有。/src/engine.cpp 可以实现这个功能。

```
#include "engine.h"#include <iostream>void Initialise(){ std::cout << "ENGINE INITIALISED!" << std::endl;}
```

很好，一个强大的游戏引擎！

现在让我们创建临时测试可执行文件将使用的文件。那是。/src/testing.cpp

```
#include "engine.h"int main(){ Initialise(); return 0;
}
```

这可能看起来非常基本和简单，但这是我们整个项目的基础！

时间到了，我们全能游戏引擎的第一次构建…希望我们没有增加太多功能！

让我们进入终端，确保我们已经将目录设置为项目的根目录，并创建一个名为 bin 的新文件夹。在 Mac 和 Linux 上，创建名为 bin 的文件夹的命令是“mkdir bin”。

通常将这个文件夹称为 bin，因为编译后的二进制文件将存放在这里。让我们进入该文件夹并运行 cmake 命令。

```
cmake --help
```

这会吐出一堆信息，但我们感兴趣的，是靠近底部的生成器列表。

```
* Unix Makefiles               = Generates standard UNIX makefiles.
  Ninja                        = Generates build.ninja files.
  Xcode                        = Generate Xcode project files.
  CodeBlocks - Ninja           = Generates CodeBlocks project files.
  CodeBlocks - Unix Makefiles  = Generates CodeBlocks project files.
  CodeLite - Ninja             = Generates CodeLite project files.
  CodeLite - Unix Makefiles    = Generates CodeLite project files.
```

这些是我的 CMake 可以生成的可能的构建系统。你的可能看起来不一样。“Unix Makefiles”旁边有一个星号，表示这是我的默认设置。这是完美的，因为这是我想用的。如果你在 Mac 或 Linux 上，那么这也是你想要的，但是如果你在 Windows 上，那么你将需要使用 Visual Studio 生成器。继续复制列表中的 Visual Studio 生成器名称。

让我们运行 CMake！

如果你在 Mac 或 Linux 上…

```
cmake ..
```

…对于 Windows，只需确保用您从列表中复制的精确文本字符串替换该命令。

```
cmake .. -G"Visual Studio 16 2019"
```

这个命令告诉 cmake 运行，但是要在下面的目录中查找 CMakeLists.txt 文件，这就是..意味着。那..可能是磁盘上某个地方的完整绝对路径，只要那里有 CMakeLists.txt 文件，它就能工作。

如果一切顺利，那么我们现在将在 bin 目录中有一个生成的项目准备构建了！在 Windows 上，打开已创建的 Visual Studio 项目，然后点击 build and run。

在 Mac 和 Linux 上，确保你进入了 bin 目录(或者任何你运行 cmake 的地方),然后简单的输入…

```
make
```

这将运行 CMake 创建的 Makefile，并构建我们的库目标和可执行文件。

如果代码中有任何语法错误，那么……修复它们……然后再试一次。

构建成功后，继续运行测试可执行文件，方法是双击它或者在终端中运行它。

```
./program
```

我们应该看到输出“引擎初始化！”

我们在做生意！功能强大的游戏引擎！

我希望你对此感到兴奋，它没有花哨的 3D 图形…或者任何图形…然而，我们现在有一个组织良好的项目，我们可以在此基础上继续发展。继续并完全删除 bin 文件夹。什么？！删除 bin 文件夹？！你一定是疯了！

我没疯！bin 文件夹的内容是由 CMake 为我们生成的，这意味着我们不必担心它，当然也不应该将其签入版本控制。因此，继续删除 bin 文件夹，重新创建它，在您的终端 cd 到它，并尝试再次生成项目。

不过，我们不需要每次进行更改时都重新生成，CMake 非常强大，它知道您何时更改了源代码或 CMakeLists.txt 文件本身。它会检测变化并为您更新自身。所以我们的开发过程将会是这样的:

1.  创建或编辑源代码文件
2.  向 CMakeLists.txt 添加任何新文件
3.  在 IDE 中运行“make”或“build”。
4.  重复直到游戏引擎完成。

我希望你能体会到 CMake 在维护你自己的 Makefile / XCode 项目/ Visual Studio 解决方案等方面的强大功能。不仅仅是 CMake 跨平台，如果我们对生成的项目乱搞，破坏了什么，我们可以重新生成！

目前就这些。下一集，我们将添加一些真正的代码到我们的引擎中。