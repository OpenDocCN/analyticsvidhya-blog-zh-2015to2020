# 使用 Git Rebase 提交 Git Squash

> 原文：<https://medium.com/analytics-vidhya/git-squash-commit-with-git-rebase-34443d271f62?source=collection_archive---------16----------------------->

![](img/04c3a4b16b09a5ca040d34bbdf783b0f.png)

当提交一个 pull 请求以将您的代码与 Master/Develop 合并时，您最好压缩您的提交。一些与 git repos 交互的应用程序将提供一个挤压用户界面。但是让我们走有趣的路线— *终点站*。

做 git 壁球有多种方法。第一，在你的系统中本地完成，然后推到远程。另一种方法是，在进行 rebase 之前，在 remote 中保存一份所有更改的副本，以便在出现问题时在 remote 中保存一份更改的副本。

让我们先看看更安全的方法。确保您的分支与远程服务器保持同步。现在做`git log --pretty=oneline`来理解在你的分支中发生的提交。

```
* c88bc5 Implement search inputs for user
* 8f4917 Enriched plots for better understanding
* 59c01d Add pyplot for better analysis
* ba6f1f Add listing feature to quality checks
* 9f2adb Add feature to pipeline
* f796c1 Initial commit
```

如果将最后 6 次提交打包在一起，看起来会更好，所以让我们通过交互式的重新调整来实现。

要交互地重置提交，您可以遵循下面的格式，并通过命令行输入您的命令。

```
git rebase -i HEAD~<n> (n is the number of commits you want to squash)git rebase -i HEAD~6 (This will roll up all 6 commits in the current branch)
```

或者

```
git rebase -i <sha code> (sha code of the commit until which you want to squash)git rebase -i f796c1 (sha code of the initial commit)
```

-i 标志表示这个重定基础过程将是一个交互式会话。

一旦您输入上述命令，您将看到以下内容。

```
pick f796c1 Initial commit
pick 9f2adb Add feature to pipeline
pick ba6f1f Add listing feature to quality checks
pick 59c01d Add pyplot for better analysis
pick 8f4917 Enriched plots for better understanding
pick c88bc5 Implement search inputs for user*# Rebase 8db7e8b..fa20af3 onto 8db7e8b* 
*#* 
*# Commands:* 
*#  p, pick = use commit* 
*#  r, reword = use commit, but edit the commit message* 
*#  e, edit = use commit, but stop for amending* 
*#  s, squash = use commit, but meld into previous commit* 
*#  f, fixup = like "squash", but discard this commit's log message* 
*#  x, exec = run command (the rest of the line) using shell* 
*#* 
*# These lines can be re-ordered; they are executed from top to bottom.* 
*#* 
*# If you remove a line here THAT COMMIT WILL BE LOST.* 
*#* 
*# However, if you remove everything, the rebase will be aborted.* 
*#* 
*# Note that empty commits are commented out*
```

我们看到最后 6 次提交，从旧的到新的。看到提交列表下面的评论了吗？解释得好，饭桶！`pick`是默认操作。在这种情况下，它将按原样重新应用提交，内容或消息不变。保存此文件不会对存储库进行任何更改。

我们只对以下行为感兴趣。

*   `squash`(简称为`s`)，它将提交合并到前一个(前一行中的那个)
*   `fixup`(简称为`f`)，其行为类似于“挤压”，但会丢弃提交消息

假设我们想要压缩所有的提交，因为它们属于同一个逻辑变更集。我们将保留初始提交，并将所有后续提交压缩到前一个提交中。除了第一次提交，我们必须在所有提交中将`pick`改为`squash`。

```
pick f796c1 Initial commit
squash 9f2adb Add feature to pipeline
squash ba6f1f Add listing feature to quality checks
squash 59c01d Add pyplot for better analysis
squash 8f4917 Enriched plots for better understanding
squash c88bc5 Implement search inputs for user
```

保存编辑器，您将进入另一个编辑器来决定合并的三个提交的提交消息。在这个编辑器中，您可以选择添加/删除提交消息。一旦保存了提交消息并退出了编辑器，所有的提交都会被转换成一个。

如果您想跳过编辑提交消息部分，您可以使用`fixup`命令，这将使您的提交消息已经被注释掉。

一旦保存了提交消息部分，您必须做的最后一件事就是 git push 将您的所有更改推送到 remote。而且这种推动是必须的，因为你的本地和远程的分支在重定基数后已经分开了。

```
git push --force
```

另外，如果你有太多的提交，要被压制，你必须手动更新每一个`pick`到`squash`，`vim`提供了一个简单的方法来实现它。

```
:%s/pick/squash/gc
```

该命令将在您确认后更新每个要挤压的拾取。

如果您在想要编辑的提交中说`reword`(简称`r`):

```
pick f796c1 Initial commit
pick 9f2adb Add feature to pipeline
reword ba6f1f Add listing feature to quality checks
pick 59c01d Add pyplot for better analysis
pick 8f4917 Enriched plots for better understanding
pick c88bc5 Implement search inputs for user*# Rebase 8db7e8b..fa20af3 onto 8db7e8b* 
*#* 
*# Commands:* 
*#  p, pick = use commit* 
*#  r, reword = use commit, but edit the commit message* 
*#  e, edit = use commit, but stop for amending* 
*#  s, squash = use commit, but meld into previous commit* 
*#  f, fixup = like "squash", but discard this commit's log message* 
*#  x, exec = run command (the rest of the line) using shell* 
*#* 
*# These lines can be re-ordered; they are executed from top to bottom.* 
*#* 
*# If you remove a line here THAT COMMIT WILL BE LOST.* 
*#* 
*# However, if you remove everything, the rebase will be aborted.* 
*#* 
*# Note that empty commits are commented out*
```

当您保存并退出编辑器时，git 将遵循 reword 命令，并将您再次带入编辑器，就像您修改了 commit `ba6f1f`一样。现在，您可以编辑提交消息，保存并退出编辑器。

如果你喜欢这篇文章，请点击👏所以其他人会在媒体上看到它。