# 从 Xcode 断点记录 Google 工作表中的开发错误。

> 原文：<https://medium.com/analytics-vidhya/log-development-bugs-in-a-google-sheet-from-an-xcode-breakpoint-7b493251aa3f?source=collection_archive---------21----------------------->

## 使用您自己的 Python(不是 LLDB 的)的高级 Xcode 调试器

Xcode 调试器是一个被称为 LLDB 或低级调试器的优秀程序，这个调试器包括一个有用但有限的 Python 版本，可以分析你的代码、脚本等等。

但是如果你想增加更多的功能会怎么样呢？

如果你需要安装一个其他人制作的模块，比如谷歌的“gspread”库，当有人…