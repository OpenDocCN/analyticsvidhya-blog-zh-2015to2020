# C 语言中的指针

> 原文：<https://medium.com/analytics-vidhya/pointers-in-c-and-c-ecb18cee6e28?source=collection_archive---------27----------------------->

![](img/e66435654ab69f5760ff309c6fe5eb25.png)

啊，古老的好指针。大多数计算机科学课程中的第一个主题，也是大多数学生都很难理解的。我承认。并不是每个人都在日常编程中使用指针。事实上，在几家公司实习期间，我从未使用过指针。然而，在学习了操作系统和几门关键的计算机科学课程后，我意识到它们都有一个共同的敌人或盟友，这取决于你的视角:**指针**。它恼人的持续出现在我的教育生涯中，迫使我重新考虑它在开发人员世界中的重要性和相关性。

尽管大多数开发人员永远不会用到它，但知道它的存在以及它在日常计算生活中的作用还是很有好处的。原因如下。你写的每个应用程序都会消耗内存。这是一个不可否认的事实，你必须接受。任何涉及内存的东西都必然会用到指针。有些人可能会说，即使是最复杂的 python 程序也没有包含一个指针。实际上，python 语言已经为您完成了这项工作，所以您不必这么做。被宠坏的混蛋😙

# 什么是指针？

指针只是一个给变量的通称，用来保存某个东西的内存地址。这里的类比是，指针是指**指向***没有双关语的意思，你到你想去的地方。

**声明指针**

```
int *int_pointer; 
char *char_pointer; 
float *float_pointer; 
struct Person *person_pointer;
```

指针的一般格式是`[Data Type] *[Variable Name]`。`*`表示该特定数据类型的变量是指针。

**初始化指针**

指针存储内存地址，但是我们如何得到某个东西的内存地址呢？c 有一个特殊的内置操作符`&`，它返回任意值的地址。

```
int value =2; 
int *location_of_value = &value;
```

这里变量`value`包含 2，我们使用`&value`在内存中检索这个“2”的位置，并将其赋给指针`*location_of_value`

## 从指针中检索值

给定一个指针，要从指针中检索值，我们必须使用`*`取消对它的引用。换句话说，一旦我们看到街道标志，我们如何前往该位置？

```
int value =2; 
int *location_of_value = &value; 
int retrieved_value=*location_of_value; // retrieved_value == 2
```

这里我们使用`*location_of_value`从位置中检索值

# 指针的特殊情况

**数组**

任何数据类型的数组都是指针用法的特例，其中与数组关联的变量是指向数组第一个元素的指针。听起来很困惑？我也是。

```
int arr[3]= {1,2,3};
printf("%p\n", arr);
```

printf 语句中的`%p`用来打印指针的地址。这里，当我们打印指针的地址并传递`arr`时，我们期望`arr`包含一个地址。这个地址存放着什么？

```
int arr[3]= {1,2,3};
printf("%d\n", *arr);// prints 1
```

当我们遵从存储在`arr`中的这个地址时，我们得到值 1，这是数组中的第一个元素。为了得到数组中的下一个值，我们只需要告诉指针移动到下一个地址。

```
int arr[3]= {1,2,3};
printf("%p\n", arr);// prints 1
arr++;
printf("%d\n", *arr); // prints 2
```

当我们增加指针时，我们告诉它移动`sizeof(int)`字节，这是数组中下一个元素的内存地址。

**琴弦**

c 语言中实际上没有字符串这种东西，字符串只是一个 char 的数组。

```
char str[10]= "Hi There\0";
 printf("%c", *str); // prints "H"
```

这里`str`指向 char 序列中第一个字符的内存地址。打印整个字符串

```
char str[10] = "Hi There \0";
while(str){
    printf("%c", *str);
    str++;
}
// prints "Hi There"
```

**类和结构**

结构和类在取消隔离的步骤上略有不同

```
typedef struct person{
    char* first_name;
    int age;

}person;
.
.
.   
char name[10] = "Bob";
person p1 = {name, 21};
person *p1_location = &p1;
printf("First name:%s age:%d \n", p1_location->first_name, (*p1_location).age);
```

给定一个 `struct person *`来访问 person 结构的成员字段，有两种方法。

1.  `p1_location->firstname`-在一个操作符中解除引用和访问
2.  `(*p1_location).firstname`-先解引用再访问

**内存分配和指针**

分配内存块时也使用指针

```
char *str_ptr = malloc(sizeof(char) * 10)
strcpy(str_ptr, "hello\0");
printf("%s\n", str_ptr) // prints "hello"
```

这里我们要求`malloc`给我们一块可以存储 10 个字符的内存，并返回给我们第一个位置的起始内存地址。

# 最后一个音符

这个例子列表并不是指针应用方式的详尽列表，但是对于那些不熟悉内存地址和指针的人来说，它是一个很好的起点。和平出去:)

*最初发表于*[https://www.devsurvival.com/pointers-with-c/](https://www.devsurvival.com/pointers-with-c/)