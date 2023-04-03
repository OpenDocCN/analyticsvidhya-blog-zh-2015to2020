# GitHub CLI 1.0:您需要知道的一切

> 原文：<https://medium.com/analytics-vidhya/github-cli-1-0-all-you-need-to-know-7cdd9d96c014?source=collection_archive---------11----------------------->

今年早些时候，GitHub 宣布了 GitHub CLI 的测试版，GitHub CLI 1.0 现已推出。

![](img/4bc31d1f8d90c58edb104c7efb46bbb4.png)

GitHub CLI 基本上就是把 GitHub 带到你的终端上。使用 GitHub CLI，开发人员可以检查 GitHub 问题和请求的状态，搜索特定问题或 PR，创建/派生 repo，或者直接从命令行创建新问题和请求。

它减少了上下文切换，帮助您集中注意力，并使您能够更轻松地编写和创建自己的工作流。

这篇博文将涵盖哪些内容:

1.  什么是 GitHub CLI
2.  如何下载和认证
3.  使用 GitHub CLI 管理 GitHub 存储库
4.  使用 GitHub CLI 处理拉请求
5.  使用 GitHub CLI 管理 GitHub 问题
6.  使用 GitHub CLI 使用 GitHub gist
7.  使用 GitHub CLI 的 GitHub 别名

# 什么是 GitHub CLI？

我们先简单介绍一下 GitHub CLI。GitHub CLI 最好被描述为“来自命令行的 GitHub”。

使用 GitHub CLI 1.0，您可以:

*   从终端运行整个 GitHub 工作流程，从发布到发布
*   调用 GitHub API 来编写几乎任何动作的脚本，并为任何命令设置自定义别名
*   除了[GitHub.com](http://github.com/)连接 GitHub 企业服务器

`gh`是 GitHub 在命令行。它将 pull 请求、问题和其他 GitHub 概念带到您已经在使用 git 和代码的终端旁边。

# 怎么下载？

要在您的机器上安装 GitHub CLI，您可以参考 GitHub 上的安装指南。要下载，请单击[链接](https://cli.github.com/)单击该链接导航至[指南](https://github.com/cli/cli#installation-and-upgrading)

GitHub CLI 适用于所有平台，MacOS，Windows 和各种 Linux。

# 证明

成功安装后，您需要验证您的 Github 帐户。

要验证您的帐户，您需要使用以下命令。

```
gh auth login
```

按照一系列步骤完成您的身份验证过程。认证之后，就可以使用 GitHub CLI 了。

# 使用 GitHub CLI 管理 GitHub 存储库

`gh repo`命令用于创建、克隆、派生和查看存储库。

# 1.查看 GitHub 存储库

显示 GitHub 存储库的描述和自述文件。如果没有参数，将显示当前目录的存储库。

```
gh repo view [<repository>] [flags]
```

在浏览器中打开存储库

您可以使用`-w, --web`命令在 web 浏览器中打开存储库。

# 2.创建 GitHub 存储库

使用以下命令创建一个新的 GitHub 存储库。

```
gh repo create [<name>] [flags]
```

例如:创建一个具有特定名称的存储库

```
$ gh repo create demo-repository
```

# 3.Fork GitHub 仓库

让我们用 GitHub CLI 派生一个存储库。

如果不提供任何参数，会创建当前存储库的一个分支。否则，派生指定的存储库。

```
gh repo fork [<repository>] [flags]
```

# 使用 GitHub CLI 处理拉请求

让我们探索一下如何使用 GitHub CLI 来管理 pull 请求。

# 1.列出拉取请求

```
gh pr list
```

如果我们想列出所有的拉请求，包括打开的和关闭的，我们可以使用“state”标志

```
gh pr list --state "all"
gh pr list -s "all"
```

当列出所有 PRs 时，不使用此完整命令

```
gh pr list --label "labelname"
```

# 2.检查拉取请求状态

如果您希望检查您之前创建的 PRs 的状态，您可以使用`status command`在终端列出它们

```
gh pr status
```

这将为您提供分配给您的、提及您的或由您打开的 PRs 列表。

如果您希望检查一个采购请求是否已关闭，请使用以下命令。

```
gh pr list --state "closed"
gh pr list -s "closed"
```

如果您希望列出我们所有打开的错误修复 PRs，您可以通过再次列出所有打开的 PRs 来进行检查

您还可以通过 GitHub repo 中的定义标签名称进行过滤。

# 3.查看拉取请求

您可以使用 view 命令从命令行打开 PR。

```
gh pr view [<number> | <url> | <branch>] [flags]
```

它将在 web 浏览器中打开“拉”请求，然后您可以将其分配给自己，对其进行检查等。

# 4.创建拉式请求

您可以使用以下命令直接从命令行创建新的拉请求。

```
gh pr create --title "Pull request title" --body "Pull request body"
```

您可以选择提交 PR，您可以在浏览器中打开它，或者取消。

如果你喜欢从网上创建你的 PR，你可以使用下面的命令打开一个浏览器窗口。

```
gh pr create --web
```

# 5.结帐拉取请求

您可以使用下面的命令在 GitHub CLI 中检查一个 Pull 请求。

```
gh pr checkout {<number> | <url> | <branch>} [flags]
```

# 使用 GitHub CLI 管理 GitHub 问题

# 1.列出 GitHub CLI 的问题

```
gh issue list
```

如果我们想列出所有的问题，我们可以使用`state`标志

```
gh issue list --state "all"
gh issue list -s "all"
```

# 2.使用 GitHub CLI 检查问题状态

您可以使用以下命令在终端列出它们。

```
gh issue status
```

这将为您提供分配给您的、提及您的或由您打开的问题列表。

万一你找不到你要找的问题，让我们检查它是否关闭。

```
gh issue list --state "closed"
gh issue list -s "closed"
```

如果你想过滤掉，你可以通过 GitHub repo 中定义的“labelname”标签进行过滤

```
gh issue list --label "labelname"
gh issue list -l "labelname"
```

# 3.使用 GitHub CLI 查看问题

您可以使用以下命令从命令行打开问题。

```
gh issue view {<number> | <url>} [flags]
```

例如:

```
gh issue view "10"
```

# 4.使用 GitHub CLI 创建问题

您可以借助以下命令创建一个问题。

```
gh issue create --title "Issue title" --body "Issue body"
```

最后，您可以选择提交问题、在浏览器中打开问题或取消。

如果您仍然喜欢从 web 创建问题，您可以使用以下命令

```
gh issue create --web
```

# 使用 GitHub CLI 使用 GitHub gist

# 1.用 GitHub CLI 列出要点

您可以使用下面的命令在 GitHub CLI 中列出您的 gist。

```
gh gist list [flags]
```

# 2.使用 GitHub CLI 查看要点

您可以通过 GitHub CLI 使用以下命令查看您的 gists。

```
gh gist view {<gist id> | <gist url>} [flags]
```

# 3.使用 GitHub CLI 创建 gist

用给定的内容创建一个新的 GitHub gist。可以从一个或多个文件创建 Gists。或者，传递“-”作为从标准输入中读取的文件名。

默认情况下，gists 是私有的；使用“–public”可以公开列出。

```
gh gist create [<filename>... | -] [flags]
```

# 使用 GitHub CLI 使用 GitHub alias

您可以使用以下命令打印出所有配置使用的别名`gh`。

```
gh alias list [flags]
```

我希望这对你有帮助。你可以在[推特](https://twitter.com/ayushi7rawat)上和我联系

另外，看看我的另一个博客:

*   [Instagram Bot](/analytics-vidhya/how-to-make-an-instagram-bot-with-python-fec28bb12e95)
*   [Python 3.9:你需要知道的一切](/@ayushi7rawat/python-3-9-all-you-need-to-know-a9f28236bf1f)
*   [网页抓取冠状病毒数据到 MS Excel](/@ayushi7rawat/web-scraping-coronavirus-data-into-ms-excel-f0bf350e8ed4)
*   [如何制作自己的谷歌 Chrome 扩展](/@ayushi7rawat/do-you-use-chrome-extensions-have-you-ever-thought-of-making-one-72984aef041e)

资源:

*   [github.blog/2020–09–17-github-cli-1–0-is-no..](https://github.blog/2020-09-17-github-cli-1-0-is-now-available/)
*   [cli.github.com/manual](https://cli.github.com/manual/)
*   [github.com/cli/cli](https://github.com/cli/cli)