# 我的 Git 小抄

> 原文：<https://medium.com/analytics-vidhya/my-git-cheat-sheet-d696f6b20400?source=collection_archive---------0----------------------->

## 把你从灾难中解救出来的命令。

![](img/e6577a68cd98a0ce2621dfa81a74b9fe.png)

> *我们在生活中推动的东西，会在永恒中回响！*

如果您发现您在上次签入时不小心遗漏了一些东西，不管是文件还是您刚刚提交的文件的额外更改，不要担心。这很容易解决。

```
git add the_poor_left_out_file.txt
git commit --amend
git push -f origin some_branch
```

这就对了。

您还可以使用–no-edit 标志允许在不更改提交消息的情况下对提交进行修改。

但是你永远不应该修改你已经推送到公共存储库的公共提交，因为修改实际上从历史中删除了最后一个提交，并且创建了一个新的提交，该提交具有来自该提交的组合的改变，并且在修改时新添加，有时将它们两者合并。

# 呀！我真的需要从上次提交中删除该文件

> 不要让墙塌下来，git 知道你在想什么。

如果您刚刚暂存了文件，但尚未提交，则只需重置暂存的文件:

```
git reset the-so-so-unwanted-file
```

如果你已经做了太多的改变，那么就这样做:

```
git reset --soft HEAD~1
git reset the-so-so-unwanted-file 
rm the-so-so-unwanted-file 
git commit
```

## 防止再次被意外添加

Git 将使用。 *gitignore* 确定哪些文件和目录不应该在版本控制下被跟踪。的。 *gitignore* 文件存储在您的存储库中，以便与任何其他与存储库交互的用户共享忽略规则。

```
git rm --cached the-file-to-ignore 
echo the-file-to-ignore >> .gitignore 
git add .gitignore 
git commit --amend --no-edit
```

很简单！

# 保持冷静，把它藏起来

> 如果你有什么计划，现在就提交或者永远藏起来。

暂时搁置(或*隐藏*)您对工作副本所做的更改，以便您可以处理其他内容，然后回来重新应用它们。如果您想要快速切换上下文并处理其他内容，但是您正在进行代码更改，并且还没有准备好提交应用程序的当前状态，那么 Stashing 就非常方便。

```
git stash Saved working directory and index state WIP on master: c584a97 .. HEAD is now at c584a97 ..
```

此时，您可以自由地进行更改、创建新的提交、切换分支和执行任何其他 Git 操作；当你准备好了，再回来重新使用你的储备。

**注意:** *存储在您的 Git 存储库中；推送时，堆栈不会传输到服务器。*

```
git stash pop Dropped refs/stash@{0} (80227cb4a12508bc75ca842466395971587ea09e)
```

或者，您可以重新应用更改。

```
git stash apply
```

如果您决定不再需要某个特定的存储，您可以使用`git stash drop`删除它

```
git log --oneline --graph stash@{0}
*   93ea909 WIP on master: c584a97 commit-message
|\  
| * 385d954 index on master: c584a97 commit-message
|/
```

# 哦，不，我不小心犯了主人，而不是一个分支

> 当你玩克隆人游戏时，你要么合并，要么重置——很难。

别担心。只需将所有更改回滚到一个新分支。**注意:** *确保你先提交或者保存你的修改，否则一切都会丢失！*

```
git branch the-awesome-feature-branch 
git reset HEAD~ --hard 
git checkout the-awesome-feature-branch 
git commit
```

这将创建一个新的分支，将主分支回滚到您进行更改之前的位置，并最终签出您的新分支，所有以前的更改保持不变。

婷婷婷婷！

# 好了，我现在可以更改提交消息了吗？

> *重写历史要小心。它可能会促使你使用原力的黑暗面。*

如果它仅存在于您的本地存储库中，并且尚未被推送，您可以通过执行以下操作来修改提交消息:

```
git commit --amend
```

如果您已经推送了提交，您将不得不使用修改后的消息强制推送提交。

```
git rebase -i HEAD~n
```

在您想要更改的每个提交消息之前，用`reword`替换`pick`。

保存并关闭提交列表文件。在每个生成的提交文件中，键入新的提交消息，保存文件并关闭它。那就施魔法吧！

```
git push --force
```

**提交消息遵循的规则:**

*   不要以句号结束提交消息。
*   将您的提交消息控制在 50 个字符以内。
*   使用主动语态。比如用“添加”代替“添加”，用“合并”代替“合并”。
*   将您的提交视为表达引入变更的意图。

# 哎呀！警告分支名称

> 小心不要移动你踩着的树枝。

如果您还没有将您的分支签入远程宇宙，那么请执行以下操作:

```
git branch -m feature-brush feature-branch
```

如果您已经推送了该分支，那么我们需要从远程删除旧分支，并推送新分支。

```
git push origin --delete feature-brush 
git push origin feature-branch
```

# 时光旅行回到过去

> 我们走的每一步，我们做的每一个动作，git 都和我们在一起，永远！

```
git clean -n
```

这类似于对未跟踪文件的**撤销**。如果情况更糟，我们总能回到过去。

```
git show [commit]
```

这将输出指定提交的元数据和内容更改。

```
$ git reset [commit]
```

这将撤消[commit]之后的所有提交，并在本地保留更改。

```
$ git reset --hard [commit]
```

这将丢弃所有历史记录，并更改回指定的提交，即*灭霸快照！*

# 刷新 git 以使用新用户

> 一天一个饭桶可以避免冲突。

```
remote: Permission to .git denied to MY_OLD_GIT_ACCOUNT fatal: unable to access 'url': The requested URL returned error: 403
```

此错误意味着您推送的密钥作为部署密钥附加到另一个存储库，并且无权访问您尝试推送的存储库。

使用详细选项-v 进行检查

```
$ git remote -v origin https-url-of-my-awesome-app.git (fetch) origin https-url-of-my-awesome-app.git (push)
```

如果你在这里看到一些不同的东西，设置一个 url

```
git remote set-url origin git@some.git.url:/.git
```

仅此而已！

好吧，我希望你喜欢 git 上的这篇文章。我仍然在探索 git 和 VCS 的许多方面。希望在不断探索中写出更多。这些技巧肯定会帮助你更好地使用 git，但是不管怎样，如果你搞砸了，

```
cd .. 
sudo rm -r i-am-hopeless-dir 
git clone [https://some.git.url/i-am-hopeless-dir.git](https://some.git.url/i-am-hopeless-dir.git) 
cd i-am-hopeless-dir
```

*原载于 2018 年 11 月 3 日*[*kuharanbhowmik.wordpress.com*](https://kuharanbhowmik.wordpress.com/2018/11/03/my-git-cheat-sheet/)*。*