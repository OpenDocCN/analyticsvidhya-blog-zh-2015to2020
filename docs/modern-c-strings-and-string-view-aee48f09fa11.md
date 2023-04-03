# 现代 C++:字符串和 string_view

> 原文：<https://medium.com/analytics-vidhya/modern-c-strings-and-string-view-aee48f09fa11?source=collection_archive---------6----------------------->

# c 样式字符串

除了与 C 库接口时，应该避免 C 风格的字符串。c #字符串库函数不提供边界检查和内存分配支持。它们被表示为字符数组。字符串的最后一个字符是空字符`\0`，因此使用该字符串的代码知道它在哪里结束。字符串所需的空间总是比可读字符数多 1。

# 字符串文字

用引号括起来的字符串是字符串文字。它们存储在存储器的只读部分。因为它们存储在只读部分，试图修改字符串文字是*未定义的行为*。示例:

```
char *str = "world";
str[0] = 'y'; //undefined behavior
```

如果代码遵循标准，并将字符串赋值给`const char*`，编译器将捕捉修改字符串的尝试:

```
const char * str = "world";
str[0] = 'k'; //compiler will flag this as error.
```

要改变字符串，将它们分配给字符数组。在此示例中，编译器创建了一个足够大的数组来保存字符串，并将字符串复制到该数组中。文本不放在只读内存中。

```
char str[] = "hello world";
str[0] = 'y';
```

# 原始字符串文字

C++提供了原始的字符串文字，这些文字跨越多行代码，不需要对嵌入的双引号进行转义，也不用任何特殊的逻辑来处理转义序列。示例:

```
const char* str = R"(Brave New World)";
const char* str = R"(Line1: Brave 
Line2: New World)";
```

要在原始字符串中使用`)`或`(`，请使用如下所示的唯一分隔符序列:

```
const char* str = R"-(Embedded ) parens)-";
```

# 标准::字符串

`string`将`+`运算符重载以表示串联。于是下面产生了`hello world`:

```
string one("hello");
string two("world");
string final;
final = one + " " + two;
```

此外，`==`、`!=`、`<`等都是重载的，所以它们使用实际的字符串字符。为了兼容，可以调用`c_str()`来返回一个 C 风格的字符串，但是这应该是最后一个操作，不应该在分配了`string`的堆栈中完成。因为字符串被推断为`const char*`，所以可以使用`s`将字符串解释为`std::string`:

```
auto myStr = "hello brave world"s;
```

# 转换策略

`std`名称空间有几个将数字转换成字符串的函数。例子:`string to_string(int vaL)`和`string to_string(double val)`。要将字符串转换为数字，请使用`int stoi(const string& s, size_t *idx = 0, int base=10)`等函数。示例:

```
const string toConvert = " 123ABC";
size_t index = 0;
int value = stoi(toConvert,&index); //index will be the index of 'A'.
```

# 标准::字符串 _ 视图

一个`string_view`是只读的`string`，但是没有一个`const string&`的开销。它不复制字符串。要将一个`string_view`和一个字符串连接起来，使用`data()`成员函数:

```
string str = "Hard";
string_view sview = " Rock";
auto result = str + sview.data();
```

要将`string_view`传递到函数中，请按值传递它们。对于`string_view`文字，使用“sv”:

```
auto sview = "Sample string view"sv;string_view extractLastDelimitedElement(string_view sv, char delimiter)
{
    return sv.substr(sv.rfind(delimiter));
}
```

# 参考资料:

m .格雷瓜尔(2018)。*专业 C++* 。印第安纳州，约翰·威利的儿子们。

[](https://codingadventures1.blogspot.com/2020/02/c-strings-and-stringview.html) [## C++字符串和 string_view

### 除了与 C 库接口时，应该避免 C 风格的字符串。c #字符串库函数不提供…

codingadventures1.blogspot.com](https://codingadventures1.blogspot.com/2020/02/c-strings-and-stringview.html)