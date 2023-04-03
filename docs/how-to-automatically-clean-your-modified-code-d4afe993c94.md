# 如何自动清理自己修改过的代码？

> 原文：<https://medium.com/analytics-vidhya/how-to-automatically-clean-your-modified-code-d4afe993c94?source=collection_archive---------34----------------------->

# 介绍

在这篇文章中，我将向你展示我的一些想法，来修复到目前为止还没有使用代码格式化程序的存储库中的 Python 代码风格。如果没有它们，文件内容可能很难阅读，尤其是由许多用户开发时。我用`git ls-files`和`git diff`告诉你在当前工作分支中查找修改文件的方法。然后，我将解释如何使用`xargs`运行几个清理命令，如`flake8`和`black`来清理代码。虽然最后一个来自 Linux 系统，但 Windows 用户能够使用它…