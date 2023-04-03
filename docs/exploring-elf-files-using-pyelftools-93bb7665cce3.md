# 使用 pyelftools 浏览 ELF 文件

> 原文：<https://medium.com/analytics-vidhya/exploring-elf-files-using-pyelftools-93bb7665cce3?source=collection_archive---------3----------------------->

# 介绍

有很多工具可以探索 [ELF 格式](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format)的可执行文件。其中大多数旨在提供从上述格式的二进制文件中提取的唯一信息。它们很棒，但有时我们需要一种通用但高度专业化的工具，允许做比标准工具更多的事情。这是 **pyelftools** 发挥作用的时刻。

在这篇文章中，我想展示一些 **pyelftools** 的用法示例。我没有展示如何使用 pyelftools 本身，也就是它的类和其他特性，因为你可以在[文档](https://github.com/eliben/pyelftools/wiki/User's-guide)和[源代码](https://github.com/eliben/pyelftools)本身中找到。相反，我专注于这个工具在特定用途上的应用。

# 先决条件

# 环境

以下信息是我的测试环境，您的可能会有所不同:

```
hedin@home:~/projects/elf$ lsb_release -a
    LSB Version:    core-11.1.0ubuntu2-noarch:security-11.1.0ubuntu2-noarch
    Distributor ID: Ubuntu
    Description:    Ubuntu 20.04.1 LTS
    Release:        20.04
    Codename:       focal
 hedin@home:~/projects/elf$ python3 --version
    Python 3.8.5
```

# 要求

本文中给出的脚本需要:

*   Python 版或更高版本。
*   pyelftools

# 装置

有些基于 Linux 的发行版不包含 python3 或 pip3。我们还需要安装 **pyelftools** 。下面的代码块是如何在基于 debian 的发行版上安装所有提到的:

```
sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install pyelftools
```

# 用法示例

好了，现在我们已经安装了 **pyelftools** 。但是接下来呢？怎么用，为什么用？我想展示一些标准 [GNU Binutils](https://interrupt.memfault.com/blog/gnu-binutils) 工具的输出，然后提供基于 pyelftools 的代码片段。所有的代码片段都可以在我的 [ELF github 库](https://github.com/Romeus/elf)中找到。

# 用于内存和磁盘表示的不同大小的段

以下引用的规范为我们提供了关于 ELF 文件中的段的信息:

> 可执行或共享目标文件的程序头表是一个结构数组，每个结构描述一个*段*或系统准备程序执行所需的其他信息。一个目标文件段包含一个或多个节。程序头仅对可执行文件和共享目标文件有意义。

从编程的角度来看，下面的结构显示了 ELF 文件标题表中的一个段的表示:

```
typedef struct {
    Elf32_Word p_type;
    Elf32_Off p_offset;
    Elf32_Addr p_vaddr;
    Elf32_Addr p_paddr;
    Elf32_Word p_filesz;
    Elf32_Word p_memsz;
    Elf32_Word p_flags;
    Elf32_Word p_align;
} Elf32_Phdr;
```

我们对该结构的两个成员感兴趣:

> p_filesz —该成员给出段的文件映像中的字节数；可能是零。
> 
> **p_memsz** —该成员给出段的内存映像中的字节数；可能是零。

现在我们看到文件和内存中的数据段大小可能不同。这取决于路段中包含的路段数量和类型、路段的线形以及其他一些原因。我们感兴趣的是寻找具有不同大小的片段。首先，让我们看看一个伟大的 **readelf** 工具的用法，它允许我们提取这些信息:

```
hedin@home:~/projects/elf$ readelf --wide --segments /bin/ps
```

我将全部输出缩减为分段信息:

```
....................................................................
    There are 13 program headers, starting at offset 64
    Program Headers:
      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
      PHDR           0x000040 0x0000000000000040 0x0000000000000040 0x0002d8 0x0002d8 R   0x8
      INTERP         0x000318 0x0000000000000318 0x0000000000000318 0x00001c 0x00001c R   0x1
          [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
      LOAD           0x000000 0x0000000000000000 0x0000000000000000 0x009a88 0x009a88 R   0x1000
      LOAD           0x00a000 0x000000000000a000 0x000000000000a000 0x00bbf1 0x00bbf1 R E 0x1000
      LOAD           0x016000 0x0000000000016000 0x0000000000016000 0x006318 0x006318 R   0x1000
      LOAD           0x01cf70 0x000000000001df70 0x000000000001df70 0x004190 0x025478 RW  0x1000
      DYNAMIC        0x020ac0 0x0000000000021ac0 0x0000000000021ac0 0x000210 0x000210 RW  0x8
      NOTE           0x000338 0x0000000000000338 0x0000000000000338 0x000020 0x000020 R   0x8
      NOTE           0x000358 0x0000000000000358 0x0000000000000358 0x000044 0x000044 R   0x4
      GNU_PROPERTY   0x000338 0x0000000000000338 0x0000000000000338 0x000020 0x000020 R   0x8
      GNU_EH_FRAME   0x019e1c 0x0000000000019e1c 0x0000000000019e1c 0x0007b4 0x0007b4 R   0x4
      GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0x10
      GNU_RELRO      0x01cf70 0x000000000001df70 0x000000000001df70 0x004090 0x004090 R   0x1
    ....................................................................
```

我们拥有所有需要的信息，但它并没有以一种让我们感到舒服的方式出现。为了找到合适的段，我们需要检查每个段并比较 **FileSiz** 和 **MemSiz** 列。让我们编写自己的基本脚本，它将向我们显示内存和磁盘上表示的不同大小的段。这个脚本的核心是一个简单的循环，它遍历 ELF 二进制文件的所有段，并呈现满足条件 **p_filesz！= p_memsz** :

```
#!/usr/bin/env python3 import sys
    from elftools.elf.elffile import ELFFile
    from elftools.elf.segments import Segment if __name__ == '__main__': if len(sys.argv) < 2:
            print("You must provide this script with an elf binary file you want to examine")
            exit(1) print(f"Segments of the file {sys.argv[1]} which size on disk and in memory differs") with open(sys.argv[1], 'rb') as elffile:
            for segment in ELFFile(elffile).iter_segments():
                if segment.header.p_filesz != segment.header.p_memsz:
                    seg_head = segment.header
                    print(f"Type: {seg_head.p_type}\nOffset: {hex(seg_head.p_offset)}\nSize in file:{hex(seg_head.p_filesz)}\nSize in memory:{hex(seg_head.p_memsz)}")
```

现在，让我们使用以下脚本来查找所需的段:

```
hedin@home:~/projects/elf$ python3 segments.py /bin/ps
    Segments of the file /bin/ps which size on disk and in memory differs
    Type: PT_LOAD
    Offset: 0x1cf70
    Size in file:0x4190
    Size in memory:0x25478
```

正如你所看到的，在一个标准的/bin/ps 工具中，只有一个段在内存和磁盘上是不同的。

# 段与段之间映射的表示

从前面的例子中，我们知道每个段可以包含许多部分。readelf 工具给了我们前者和后者之间的映射。我减少了映射本身的输出，因为有很多信息不依赖于映射。看起来是这样的:

```
hedin@home:~/projects/elf$ readelf — segments — wide /bin/psSection to Segment mapping:
      Segment Sections...
       00     
       01     .interp 
       02     .interp .note.gnu.property .note.gnu.build-id .note.ABI-tag .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt 
       03     .init .plt .plt.got .plt.sec .text .fini 
       04     .rodata .eh_frame_hdr .eh_frame 
       05     .init_array .fini_array .data.rel.ro .dynamic .got .data .bss 
       06     .dynamic 
       07     .note.gnu.property 
       08     .note.gnu.build-id .note.ABI-tag 
       09     .note.gnu.property 
       10     .eh_frame_hdr 
       11     
       12     .init_array .fini_array .data.rel.ro .dynamic .got
```

嗯……信息量不大，也不方便用户使用，是吗？为了解决这种情况，我创建了一个小脚本，以更合适的形式显示映射。但是在我提供这个脚本之前，我将分享 elf 规范中的节结构[的定义:](http://refspecs.linuxbase.org/elf/elf.pdf)

```
typedef struct {
    Elf32_Word sh_name;
    Elf32_Word sh_type;
    Elf32_Word sh_flags;
    Elf32_Addr sh_addr;
    Elf32_Off sh_offset;
    Elf32_Word sh_size;
    Elf32_Word sh_link;
    Elf32_Word sh_info;
    Elf32_Word sh_addralign;
    Elf32_Word sh_entsize;
} Elf32_Shdr;
```

这个结构向我们展示了应该通过它的名字来识别这个部分。此外，我想强调一个更有趣的成员:

> **sh_addr** —如果该段将出现在进程的内存映像中，该成员给出该段的第一个字节应该驻留的地址。否则，该成员包含 0。

我认为脚本输出中的节外观应该由节名及其地址来标识，如下所示( **section_name** ， **sh_addr** )。

剧本的行为是什么？很简单。循环通过段，对于每个段，循环通过属于它的部分:

```
#!/usr/bin/env python3 import sys
    from elftools.elf.elffile import ELFFile if __name__ == '__main__': if len(sys.argv) < 2:
            print("You must provide this script with an elf binary file you want to examine")
            exit(1) print(f"Mapping between segments and sections in the file {sys.argv[1]}") elffile = ELFFile(open(sys.argv[1], 'rb')) segments = list()
        for segment_idx in range(elffile.num_segments()):
            segments.insert(segment_idx, dict())
            segments[segment_idx]['segment'] = elffile.get_segment(segment_idx)
            segments[segment_idx]['sections'] = list() for section_idx in range(elffile.num_sections()):
            section = elffile.get_section(section_idx)
            for segment in segments:
                if segment['segment'].section_in_segment(section):
                    segment['sections'].append(section) for segment in segments:
            seg_head = segment['segment'].header
            print("Segment:")
            print(f"Type: {seg_head.p_type}\nOffset: {hex(seg_head.p_offset)}\nVirtual address: {hex(seg_head.p_vaddr)}\nPhysical address: {(seg_head.p_paddr)}\nSize in file: {hex(seg_head.p_filesz)}\nSize in memory: {hex(seg_head.p_memsz)}\n") if segment['sections']:
                print("Segment's sections:")
                print([(section.name, hex(section['sh_addr'])) for section in segment['sections']], sep=', ', end='\n')
            else:
                print('Segment contains no sections') print('\n--------------------------------------------------------------------------------')
```

以下是这个脚本的输出(为了减小大小，我截掉了大部分片段):

```
hedin@home:~/projects/elf$ python3 segments_sections.py /bin/ps Mapping between segments and sections in the file /bin/ps
    Segment:
    Type: PT_PHDR
    Offset: 0x40
    Virtual address: 0x40
    Physical address: 64
    Size in file: 0x2d8
    Size in memory: 0x2d8 Segment contains no sections    ----------------------------------------------------------------
    Segment:
    Type: PT_INTERP
    Offset: 0x318
    Virtual address: 0x318
    Physical address: 792
    Size in file: 0x1c
    Size in memory: 0x1c Segment's sections:
    [('.interp', '0x318')] ----------------------------------------------------------------
    Segment:
    Type: PT_LOAD
    Offset: 0x0
    Virtual address: 0x0
    Physical address: 0
    Size in file: 0x9a88
    Size in memory: 0x9a88 Segment's sections:
    [('', '0x0'), ('.interp', '0x318'), ('.note.gnu.property', '0x338'), ('.note.gnu.build-id', '0x358'), ('.note.ABI-tag', '0x37c'), ('.gnu.hash', '0x3a0'), ('.dynsym', '0x3e8'), ('.dynstr', '0xdc0'), ('.gnu.version'
    , '0x121e'), ('.gnu.version_r', '0x12f0'), ('.rela.dyn', '0x13a0'), ('.rela.plt', '0x91e8')]    ----------------------------------------------------------------
    ....................................................................
```

# 不驻留在内存中的部分

我的最后一个例子展示了一些没有加载到内存中的特殊部分，也就是它们的 **sh_addr == 0** :

```
#!/usr/bin/env python3 import sys
    from elftools.elf.elffile import ELFFile if __name__ == '__main__': if len(sys.argv) < 2:
            print("You must provide this script with an elf binary file you want to examine")
            exit(1) print(f"Sections of the file {sys.argv[1]} that are not loaded into memory") with open(sys.argv[1], 'rb') as elffile:
            for section in ELFFile(elffile).iter_sections():
                if not section.header.sh_addr:
                    print(section.name)
```

这是 bin/ps 的输出:

```
hedin@home:~/projects/elf$ python3 sections_not_in_memory.py /bin/ps
    Sections of the file /bin/ps that are not loaded into memory .gnu_debuglink
    .shstrtab
```

# 摘要

**pyelftools** 是一个非常灵活方便的观测 ELF 双星的工具。它的范围远远超出了本文给出的简单示例，并允许创建成熟的探索工具。

# 参考

1.  [可执行可链接格式](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format)
2.  [pyelftools](https://github.com/eliben/pyelftools)
3.  [GNU Binutils](https://interrupt.memfault.com/blog/gnu-binutils)
4.  [ELF 脚本库](https://github.com/Romeus/elf)