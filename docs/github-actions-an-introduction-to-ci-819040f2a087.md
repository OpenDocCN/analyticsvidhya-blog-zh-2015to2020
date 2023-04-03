# GitHub Actions:CI 简介

> 原文：<https://medium.com/analytics-vidhya/github-actions-an-introduction-to-ci-819040f2a087?source=collection_archive---------6----------------------->

根据 [CC BY-SA 4.0 许可证](https://creativecommons.org/licenses/by-sa/4.0/)获得许可

![](img/963322675597285be2c5076fa51d070d.png)

在 CC0 许可下，Pixabay 在[像素](https://www.pexels.com/photo/mirror-lake-reflecting-wooden-house-in-middle-of-lake-overlooking-mountain-ranges-147411/)上的封面图片

***注意:*** *我将在我从第三方来源引用的一些文本后使用方括号(例如[1])。这些资源的链接在最后。*

# CI？那是什么？

CI 代表持续集成，不，这不是微积分 1，持续集成是指将一个开发团队所做的工作集成或合并到主要的源代码或主线或主干中。[1]基本上，它是一个工具，在开源项目的情况下，它允许一组开发人员或贡献者在相同的代码基础上同时工作和协作。一些 CI 服务——比如 GitHub——甚至为项目的维护者提供工具，让他们接受这些变更，提出改进代码的建议，甚至完全拒绝变更。它本质上允许一个大型团队在同一个项目上协作。

大多数 CI 服务需要用户制作代码基础或主线的本地副本，独立工作，并将他们的更改作为建议提交给代码主线。这些建议然后由同事或主管审查，他们或者接受，或者拒绝，或者建议对已更改的代码进行更改。随着审查变更过程的进行，一些变更被接受到主线代码中，开发人员的本地副本慢慢地不再反映实际的主线，这需要开发人员刷新本地副本。不更新本地副本通常会导致冲突，希望维护人员和开发人员能够解决这个问题。

在这篇博文中，我们将从 GitHub 和 GitHub Actions 的角度分析 CI/CD 的一些优点和缺点，以及在 Julia 开发中使用 CI/CD 的方法。

# CD 是什么？

CD 是 CI 最好的朋友；这两者通常是成对的，并且一起使用来指代软件开发的增量式风格。

使用缩写 CD 实际上引入了一点模糊性；CD 可以代表持续交付和持续部署。它们听起来可能一样，但它们之间有细微的差别。首先，我们将讨论它们的共同点。持续部署和持续交付都是在小而短的周期内生产软件，并对代码库进行增量更改，如添加功能、修复 bug 和改善用户体验，而不是每年左右发布一个与前任完全不同的重大返工产品。

两者指的是快速开发(不要与敏捷混淆)，测试和部署周期通常持续几个月。一些著名的项目如 Windows 10[2]，Linux 发行版(如 Arch Linux，Gentoo Linux 等。[3])，大多数开源包都使用这种形式的开发。

然而，连续部署和连续交付之间的主要区别在于，连续部署在发布更新之前使用自动化系统来构建、部署和测试软件，而在连续交付中同样是手动完成的。

在这篇博文中，我们将只讨论 GitHub 动作上下文中的持续部署，以及在现有 repo 上实现它的方法。因此，此后术语 CD 将代表连续部署。

# 为什么选择 CI/CD？

CI 的优势是显而易见的:整个开发团队可以无缝地同时处理相同的代码。在这篇博文中，我们将讨论局限于 GitHub 虽然可能有更好的 CI 系统可以更好地处理合并冲突，但我们将讨论作为主流服务的 GitHub，以及 GitHub 操作如何有利于开发团队的工作流程。

然而，CD 也有以下限制[4]:

*   **客户偏好:**如果客户不喜欢推送有时可能很关键的小增量更新(如安全补丁)，并且不更新他们的系统，则系统可能会受到危害，并且错误会影响系统本身的功能。例如，关键任务软件正在运行，或者该组织的工作流不支持 CI/CD。然后，一些系统可能无法更改系统文件或在更新后重新启动。
    由于这一点，公司已经采取了自己的对策来应对这一劣势:Ubuntu 每两年发布一次 LTS(长期支持)版本，在发布前进行广泛的调试，预计该软件将在未来几年内运行并得到支持。Windows 10 为不同类型的客户提供了不同的更新渠道。企业用户在发布 6 个月后会收到更新，在此期间，许多安全补丁、错误修复、用户体验改进和优化会被推送到企业版。此外，建议企业在将更新推送到该组织中的所有 PC 之前，在选定的机器上测试新的更新。
*   **需要人工干预的测试:**在医学研究领域，需要进行数年的测试和认证，所有这些繁文缛节减缓了这一过程，这样的开发周期在这些领域并不真正适用。

# 朱莉娅和 CI/CD

当然，就像其他编程语言一样，你可以在 GitHub 上建立一个代码库，你可以在这里找到这个。我们不会深入探讨这个问题，因为这篇博文假设你以前有过设置 GitHub 回购、克隆回购、创建问题、拉请求、派生回购以及与多个分支机构合作的经验。

一旦你有了一个包含 Julia 代码的 repo，用 GitHub 动作设置 CD 工作流就非常容易了。GitHub 动作本质上是每次事件发生时执行给定步骤的脚本。事件可以是时间的流逝、问题或拉取请求的创建或关于相同内容的评论的创建、提交或在此处[指定的任何其他事件](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/events-that-trigger-workflows#webhook-events)。用户可以根据需要定义步骤。它可能是运行一个脚本，测试一个构建，创建一个新的发布，标记问题和拉请求，检查拉请求中变更的可行性，等等。

对于 Julia 开发，在撰写本文时有两个 GitHub 操作:

*   设置 Julia 环境以运行 Julia 脚本:[设置 Julia 环境](https://github.com/marketplace/actions/setup-julia-environment)
*   为 Julia 包的发布添加标签: [Julia TagBot](https://github.com/marketplace/actions/julia-tagbot)

除非你正在维护和编写一个 Julia 包，否则你很少有机会需要了解第二个包。现在让我们来看看如何开始在 Julia repos 上使用 GitHub Actions。稍后，我们将了解为包设置 CI 工作流的过程。

# 为在 GitHub 上运行 Julia 脚本设置 CI 脚本

在设置包含要运行的. jl 文件的 repo 后，执行以下步骤来自动运行该脚本:

1.  在 repo 的根目录下创建一个`.github`文件夹，并在刚刚创建的`.github`文件夹中创建一个`workflows`文件夹。
2.  在这个工作流文件夹中创建一个`name.yml`文件，其中`name`是您想要为自动运行的脚本设置的名称。
3.  回到 repo 的根目录，创建一个名为`src`的文件夹。把你所有的 Julia 代码都放到这个文件夹里。现在，您的文件夹应该具有以下层次结构:

```
Repo_name
 |
 | — → src
 |      ├ — -> mycode.jl
 |      └ → morecode.jl
 | — → .github
 |        └ — — -> workflows
 └ ...                 └ — → name.yml
Note: The … stands for all the other files (README, LICENSE, and all other unimportant files)
```

4.既然代码的框架结构已经准备好了，让我们开始编写 name.yml 脚本，它将自动执行我们的脚本。回到 name.yml，我们将在那里添加一些代码来实现自动化。

5.前往[https://github . com/market place/actions/setup-Julia-environment](https://github.com/marketplace/actions/setup-julia-environment)，点击“使用最新版本”。复制出现在对话框中的代码。如果您在编写自动化脚本时拷贝了任何其他文本，请将其粘贴到记事本中。现在，让我们开始写剧本。我们首先需要定义自动化脚本的名称。为此，您可以在脚本顶部键入以下内容:

```
name: ‘Name of your script’
description: ‘Description of what your script does’
author: ‘Your name’
```

在此之下，我们将设置运行任务的时间表。

```
on:
  schedule:
    - cron: ‘*/5 * * * *’
```

上述脚本每 5 分钟运行一次。要更改此时间表或设置脚本在不同的事件上运行，请参见本文中的[。](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/configuring-a-workflow#triggering-a-workflow-with-events)

现在，让我们让脚本运行我们的 Julia 代码。为此，我们将创建一个`job`，并将`steps`添加到那个`job`中。由于运行我们的代码本质上是一个构建活动，我们将把我们的步骤放在一个`build`中。

`runs-on`只是脚本在哪个操作系统上运行的规范。这可能是这里[提到的任何操作系统](https://help.github.com/en/actions/automating-your-workflow-with-github-actions/virtual-environments-for-github-hosted-runners#supported-runners-and-hardware-resources)。我用过`ubuntu-latest`。您还需要使用我们之前从对话框中复制的代码来设置 Julia。

我们将使用构建步骤构建包(我们的 Julia 代码)，运行步骤将执行我们的 Julia 脚本。将代码粘贴到我们之前粘贴的`on`块下面。

```
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Setup Julia environment
      uses: julia-actions/setup-julia@v1.0.2
      - name: Build
      uses: julia-actions/julia-buildpkg@master
      - name: Run
      run: julia --project src/mycode.jl
```

请注意，您必须在最后一行中更改脚本的名称，这样才能正常工作。

现在，我们需要为我们的项目设置一些依赖项(Julia 代码)。由于 GitHub Actions 虚拟机没有安装所需的 Julia 包，我们必须安装它们。你可以通过简单地改变

`*using* Example, Example1`

到

```
*import* PkgPkg.add(“Example”); *using* Example
Pkg.add(“Example1”); *using* Example1
```

这将安装每次运行代码时所需的最新版本的包。

这样，我们就成功地自动化了我们的脚本！

# Julia 软件包和 CI/CD

Julia 包作为包的注册表保存在一个地方，其中有关于包的信息。每次使用 JuliaRegistrator 应用程序更新注册表时，都需要在 Julia 包的实际 repo 上生成新的版本。TagBot 自动更新软件包 repo 上的版本。

要设置 TagBot，请执行以下步骤:

1.  在您的 repo 的根目录下创建一个`.github`文件夹。
2.  在`.github`文件夹中创建一个`workflows`文件夹
3.  在`workflows`文件夹中创建一个名为`TagBot.yml`的文件。
4.  将以下代码粘贴到`TagBot.yml`文件中:

```
name: TagBoton:
  schedule:
    - cron: 0 * * * *jobs:
  TagBot:
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
```

现在，每次你在 JuliaRegistry 上为你的包创建一个发布，它会自动更新 repo 来反映最新的发布。

# 在合并拉取请求之前测试代码

有时，我们需要在合并请求之前运行测试，看看构建是否失败。这可以使用矩阵测试来完成，以便保持跨平台兼容性。为此，我们必须编写一个测试脚本来测试 repo 上的所有代码，并检查它是否按预期运行。编写`tests.jl`或`runtests.jl`脚本的过程对于每个回购来说都是不同的，所以你必须自己找出答案。

假设已经编写了测试脚本，并且可以测试整个回购的错误，我们现在可以为回购设置一个自动化的工作流。要设置自动化跨平台测试[5]:

将测试脚本(`tests.jl`或`runtests.jl`)放在存储库的根目录下。

在 repo 的根目录中创建一个`.github`文件夹，并在这个`.github`文件夹中创建一个`workflows`文件夹。

在这个`workflows`文件夹中创建一个`testing.yml`文件。

现在，让我们编写`testing.yml`的代码

首先，我们需要在脚本的顶部定义动作的名称，如下所示:

```
name: Run tests
```

现在，我们需要安排这次测试。在大多数测试场景中，您会希望在每个 push to master 和每个 Pull 请求上运行这个程序:

```
on:
  push:
    branches: master
  pull_request:
```

让我们最后为我们的脚本设置`job`。由于这是一个测试活动，我们将把所有 or 步骤放在一个`test`块中。为了支持跨平台测试，我们使用一种叫做矩阵测试的东西。这样可以在不同的操作系统和 Julia 版本上运行代码，以确保跨平台兼容性和向后兼容性。

```
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        julia-version: [‘1.0’, ‘1.1’, ‘1.2’, ‘1.3’, nightly]
        os: [ubuntu-latest, windows-latest, macOS-latest]
    steps:
      - uses: actions/checkout@v1.0.0 - name: “Set up Julia ${{ matrix.julia-version }}”
        uses: julia-actions/setup-julia@v1
        with:
          version: ${{ matrix.julia-version }} - name: Test exercises
       run: julia --color=yes --check-bounds=yes --project -e using Pkg; Pkg.test(coverage=true)
```

注意我们如何使用一个`matrix`将`runs-on`设置到不同的操作系统。现在让我们看一下步骤:首先，我们签出我们的 repo，以便我们的工作流可以访问它，然后，我们用 matrix 指定的版本设置一个 Julia 环境，最后，我们运行测试我们的包的测试脚本。

这就完成了跨平台测试的工作流程。

出于安全原因，建议在 GitHub 动作中检查我们的 repo 时使用提交散列而不是版本。因此，我们的结帐步骤将是

```
steps:
  - uses: actions/checkout@*commit hash*
  - name: “Set up Julia ${{ matrix.julia-version }}”
```

现在，该操作将按照使用该提交哈希提交后的方式使用代码。有关使用该版本如何会带来安全风险的更多信息，你可以参考[这篇博文](https://julienrenaux.fr/2019/12/20/github-actions-security-risk/)。

# 参考

由 Pixabay 在[像素](https://www.pexels.com/photo/mirror-lake-reflecting-wooden-house-in-middle-of-lake-overlooking-mountain-ranges-147411/)上制作的封面图片

[1]—[持续集成的定义来自维基百科](https://en.wikipedia.org/wiki/Continuous_integration)

[2] — [Windows 即服务，以 CI/CD 为例](https://docs.microsoft.com/en-us/windows/deployment/update/waas-overview)

[3] — [Linux 作为 CI/CD 的一个例子](https://en.wikipedia.org/wiki/Rolling_release)

[4]—[CD 的障碍](https://en.wikipedia.org/wiki/Continuous_delivery#Benefits_and_obstacles)

[5] — [包的标准测试脚本](https://github.com/invenia/PkgTemplates.jl/blob/master/test/fixtures/AllPlugins/.github/workflows/ci.yml)