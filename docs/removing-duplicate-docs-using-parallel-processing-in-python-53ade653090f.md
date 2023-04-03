# 使用 Python 中的散列和并行处理删除重复文件

> 原文：<https://medium.com/analytics-vidhya/removing-duplicate-docs-using-parallel-processing-in-python-53ade653090f?source=collection_archive---------13----------------------->

特别是在处理原始源文件时，我们发现数据的一个重要问题是重复，我们也在寻找快速有效的方法来消除重复文件。

![](img/2d2d6384d8e4a1ccc7a2a2a5253a13f6.png)

这里有一个这样的方法，如何使用哈希和并行处理快速有效地消除重复的文档。本文详细介绍了:在顺序和并行实现中使用散列法删除重复文档

## **方法是:**

1.计算所有文件的哈希值 2。使用哈希值识别唯一文件。3.删除重复的文件。

请在下面找到重要功能的详细信息

## 1.计算哈希值

这个函数将文件路径作为输入。它返回每个文件的哈希值作为输出。我目前在这里使用 md5 散列算法。您可以使用自己选择的任何其他哈希算法。
你也可以发送块大小作为参数。例如，对于大文件，您只想计算数据的前几个字节的哈希值，而不是整个文件。在这种情况下，您可以使用块大小。将值设置为 blocksize 时，请确保在下面的函数中用 file.read(block_size)替换 file.read()。

```
def calculate_hash_val(path, block_size=''):
    file = open(path, 'rb')
    hasher = hashlib.md5()
    data = file.read()
    while len(data) > 0:
        hasher.update(data)
        data = file.read()
    file.close()
    return hasher.hexdigest()
```

**2。向字典添加独特的文件**

这个函数将一个空字典和一个包含所有输入文件的字典作为键，并将它们各自的哈希值(我们使用上面的 calculate_hash_val 函数计算出来的)作为它们的值。该函数返回没有任何重复的 dic_unique

```
def find_unique_files(dic_unique, dict1):
    for key in dict1.keys():
        if key not in dic_unique:
            dic_unique[key] = dict1[key]
```

**3。从源文件中删除重复文件**

识别出唯一文件后，最后一步是删除剩余的重复文件。下面的函数用于从输入文件夹中删除重复项。它有两个输入 all_inps 和 unique_inps，分别包含文件路径和哈希值。

```
def remove_duplicate_files(all_inps ,unique_inps):
    for file_name in all_inps.keys():
      if all_inps[file_name] in unique_inps and file_name!=unique_inps[all_inps[file_name]]:
            os.remove(file_name)
        elif all_inps[file_name] not in unique_inps:
            os.remove(file_name)
```

**所需进口和申报**

请将完整的输入文件夹路径分配给“输入文件路径”变量

```
import datetime, os, sys, logging, hashlib
from pathlib import Path
from os import listdir
from os.path import isfile, join

input_files_path = r'H:\files\input'
input_files = [f for f in listdir(input_files_path) if isfile(join(input_files_path, f))]
input_files = [os.path.join(input_files_path, x) for x in input_files]
inp_dups = {}
unique_inps = {}
```

我们将把上述函数同样用于顺序和并行实现。请分别在下面找到这两种方法的代码。

## **方法 1(顺序实施)**

```
def rmv_dup_process(input_files):
    all_inps={}
    for file_path in input_files:
        if Path(file_path).exists():
           files_hash = calculate_hash_val(file_path)
           inp_dups[files_hash]=file_path
           all_inps[file_path] = files_hash
        else:
            print('%s is not a valid path, please verify' % file_path)
            sys.exit()

    find_unique_files(unique_inps, inp_dups)
    print(inp_dups)
    remove_duplicate_files(all_inps, unique_inps)
if __name__ == '__main__':
    datetime1 = datetime.datetime.now()
    rmv_dup_process(input_files)
    datetime2 = datetime.datetime.now()

    print( "processed in",str(datetime2 - datetime1))
```

## 方法 2(并行实施)

这也可以使用并行处理来实现，以使过程运行得更快，如果您想使用并行处理，只需用下面的代码替换上面的顺序实现逻辑:

```
#If you are using multiprocessing, you can also update the number of #processes in the file. The default value I used is 4
if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=4)
    keys_dict = pool.map(calculate_hash_val, input_files)
    pool.close()

    inp_dups = dict(zip(keys_dict, input_files))
    all_inps = dict(zip(input_files, keys_dict))

    find_unique_files(unique_inps, inp_dups)
    remove_duplicate_files(all_inps, unique_inps)
```

就是这样。源代码请在下面找到 github 链接。在接下来的几天里，我还将在 github repo 中添加更多并行处理的实现。

[https://github . com/KiranKumarChilla/Removing-Duplicate-Docs-Using-Hashing-in-Python](https://github.com/KiranKumarChilla/Removing-Duplicate-Docs-Using-Hashing-in-Python)