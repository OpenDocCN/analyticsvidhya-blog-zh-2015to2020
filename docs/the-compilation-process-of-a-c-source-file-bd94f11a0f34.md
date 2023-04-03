# 一个 c 源文件的编译过程

> 原文：<https://medium.com/analytics-vidhya/the-compilation-process-of-a-c-source-file-bd94f11a0f34?source=collection_archive---------16----------------------->

将一个`C`源文件编译成一个可执行程序，需要多个步骤。它们如下:

# 源文件的准备

编译`C`源文件的第一步是预处理源文件的准备。

准备步骤的第一步是物理源文件，字符，被 ***映射*** 到[源字符集](https://twiserandom.com/c/c-source-execution-basic-and-extended-character-sets/)，因此多字节编码或其他编码被映射到源字符集。

接下来， ***三个字母被它们所代表的字符代替*** 。[三字符](https://difyel.com/c/lexical/what-is-a-trigraph-in-c/)由两个询问标记和一个字符组成，用于替换某些字符。例如`??(`可以作为`[`的替代品。

最后，任何 ***反斜杠后跟一个新行*** ，都被删除。一个反斜杠后跟一个新行，可以用来在多行上写一个预处理指令，比如`#define`。

# 预处理

源文件现在由字符序列和空格组成。这些字符序列中，有些被 ***认为是*** 预处理标记，有些是注释，三分之一与预处理无关。

接下来发生的是，每个注释*，都被一个空格所代替。*

*之后，*预处理器令牌被解释为*。执行指令如`#ifdef`，扩展宏如`#define x 1`。最后，执行`#include`指令，使引用的头文件或源文件首先像第一步一样为预处理做准备，然后像第二步一样进行预处理。*

*一旦预处理完成， ***预处理工件*** 被删除。*

*预处理步骤，可以单独执行，通过发出 ***命令*** :*

```
*$ gcc -E source.c > name_of_preprocessed_file.i 
# If using the gcc compiler . $ cc -E source.c > name_of_preprocessed_file.i 
# If using the cc compiler . $ cpp -E source.c > name_of_preprocessed_file.i 
# If using the c preprocessor .*
```

*例如，这是一个`C`源文件:*

```
*/* This is a comment */
#define x 0
int y = 1,/* Comments are replaced by a single space*/y;int z = x*
```

*这是预处理这个文件的输出:*

```
*$ gcc -E source.c
int y = 1, y;int z = 0*
```

*`$ gcc -E source.c`，预处理`source.c`文件，并输出其内容。注释被替换为一个空格，并且执行预处理器指令。不执行`C`语法检查。*

# *为执行环境做准备*

*第三步，为执行环境做好准备。字符常量和字符串文字，从源字符集 ***翻译*** ，到[执行字符集](https://twiserandom.com/c/c-source-execution-basic-and-extended-character-sets/)，包括任何转义序列，如`\n`。*

*相邻的字符串文字，比如`"a" "b"`被*连接*成一个。*

*这个步骤产生的文件称为*翻译单元*。*

# *转化为装配*

*前三个步骤产生的文件称为翻译单元，由标记和空白组成。*

*根据 C 标准，对标记进行语法和语义分析。高层次的 C 语言，被翻译成低层次的*汇编语言。**

**每个 cpu 架构都可以有自己的汇编语言，例如`x64`汇编或`arm`汇编。**

**因此，在编译时，可以指定目标架构环境。**

**编译到一个架构，不同于编译器运行的架构，叫做 ***交叉编译*** 。**

**可以通过发出以下命令来执行向汇编步骤的转换:**

```
**$ gcc -S source.c -o name_of_preprocessed_file.s
# If using the gcc compiler .$ cc -S source.c -o name_of_preprocessed_file.s
# If using the cc compiler .**
```

**例如，下面的源文件:**

```
**int main(void){ 
        int x =0; 
}**
```

**被转换为程序集:**

```
**$ cc -S source.c
# Translate source.c into source.s$ cat source.s
# output the content of source.s.section        __TEXT,__text,regular,pure_instructions
        .macosx_version_min 10, 12
        .globl  _main
        .p2align        4, 0x90
_main:                                  ## [@main](http://twitter.com/main)
        .cfi_startproc
## BB#0:
        pushq   %rbp
Lcfi0:
        .cfi_def_cfa_offset 16
Lcfi1:
        .cfi_offset %rbp, -16
        movq    %rsp, %rbp
Lcfi2:
        .cfi_def_cfa_register %rbp
        xorl    %eax, %eax
        movl    $0, -4(%rbp)
        popq    %rbp
        retq
        .cfi_endproc.subsections_via_symbols**
```

# **装配**

**在这一步中，生成的汇编语言，被 ***映射为*** 。机器语言仅由`0`和`1`组成，因此源文件现在被翻译成`0`和`1`。**

**这个步骤产生的文件称为 ***目标代码*** 。目标代码还不可执行。**

**装配步骤 ***可通过发出以下命令*** 来执行:**

```
**$ as -c source.s -o source.o
# If using as , assemble an 
# assembly file into an 
# object file .$ gcc -c source.c -o source.o
# If using gcc  , translate 
# a source.c file into 
# object code .$ cc -c source.c -o source.o
# If using cc , translate a 
# source.c file into
# object code .**
```

# **连接**

**在该步骤中，从目标代码文件创建一个 ***可执行文件*** 。组合多个目标代码文件，合并部分静态库，并解析外部引用。每个操作系统都有自己的可执行目标代码格式。**

**可以通过使用`ld`命令，或者通过为`gcc`或`cc`提供选项来执行链接。例如，下面的源文件:**

```
**/*source.c file */
#include<math.h>
int main(void){
  double number = sqrt(2.9);
}**
```

**可以使用以下方法转换为目标代码:**

```
**$ gcc -c source.c**
```

**目标代码可以静态链接到`C`数学库，并通过发出命令变成可执行文件:**

```
**$ gcc source.o -lm -o executable_file_name**
```

# **最终注释**

**编译器可以一次完成所有这些步骤。例如发出`gcc source.c`或`cc source.c`，源文件被翻译成可执行文件。多个源文件，可以传递给`gcc`或`cc`。**

***原载于 2020 年 11 月 18 日 https://twiserandom.com**的* [*。*](https://twiserandom.com/c/the-compilation-process-of-a-c-source-file/)**