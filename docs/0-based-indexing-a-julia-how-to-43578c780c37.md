# 基于 0 的索引:Julia 指南

> 原文：<https://medium.com/analytics-vidhya/0-based-indexing-a-julia-how-to-43578c780c37?source=collection_archive---------4----------------------->

![](img/ec0aa274f40556ce4f6e715d721a72e2.png)

丹尼·米勒在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 介绍

与大多数语言不同，Julia 是从 1 开始的，而不是从 0 开始的。这意味着数组从索引 1 开始。这导致了明显不同的算法和循环设计。要了解每个系统的优点和一些用例，请参考本文档:[数组索引:0 vs 1](https://docs.google.com/document/d/11ZKaR0a6hvc6xmYLfmslAAPnkVRSZFGz5GZYRNmxmsQ/edit?usp=sharing) 。

在本文中，我们将探索一种方法来改变 Julia 中数组的索引，以满足我们的需要。我们将使用 OffsetArrays.jl 包来做同样的事情。它的 GitHub repo 可以在[这里](https://github.com/JuliaArrays/OffsetArrays.jl)找到。

使用 Julia 内置的示例，我们将逐步完成在 Julia 中创建自定义索引数组的过程。

# 安装软件包

在您的 REPL 中，(通过在命令行上键入`julia`来访问，并且只有当 Julia 在您的路径上时才起作用)键入:

```
*import* Pkg
```

然后，您可以使用以下命令安装`OffsetArrays`包:

```
Pkg.add(“OffsetArrays”)
```

# 导入包

要导入包，只需输入

```
*using* OffsetArrays
```

这个包现在已经导入，可以使用了。

# 基本用法

要在导入包后创建自定义索引数组，请键入:

```
array_name = OffsetVector([element1, element2, element3, element3], start_index:end_index)
```

其中 element1，2，3 等是数组的元素，`start_index`是数组第一个元素的索引，`end_index` 是最后一个元素的索引。

我们甚至可以使用这个包创建多维数组，它有这样的自定义索引:

```
array_name = OffsetArray{Float64}(undef, -1:1, -7:7)
```

这创建了一个 2D 数组，它有 3 行(索引分别为-1、0 和 1)，15 列(索引为-7、-6、-5 等等，直到 7)，包含所有的`#undef`对象。

# 示例:循环列表

让我们实现一个循环链表，而不像循环链表那样使用任何指针。我们将不会创建一个对象，我们将只写一个循环访问一个基于 0 的数组的函数。

代码有注释并解释了我们的算法:

```
*using* OffsetArrays*# Create a 0-indexed array* a = OffsetVector([‘a’, ‘b’, ‘c’, ‘d’], 0:3)*# 1-based array to compare with and show that it does not work* b = [‘a’, ‘b’, ‘c’, ‘d’]*# Create a function that takes in an array, and cycles through it cycle_count times* function print_circular(array, cycle_count)*# Loop through values of j ranging from 0 to size_of_array*cycle_count-1\. This makes j complete the correct number of cycles**# NOTE: it says size(array)[1] because the size function returns a tuple with the 2D dimensions of the array.**# Since we are working with a 1D vector, we are only interested in its x-dimension. This happens to be the 1st element. The tuple returned is 1-based since the core functions of Julia still use 1-based indexing. We have only created a custom array. We have not changed the behaviour of Julia.* *for* j in 0:(size(array)[1]*cycle_count)-1 *# Print the element at the j%size th position. This is what gives us the cyclic behaviour. As j increases to any number, the element accessed is the j%size th element. So, j snaps back to 0 when j becomes a multiple of the size. This means that at the end of a cycle, the 0th element is accessed. This is the reason 0-based indexing is necessary.* print(array[j%size(array)[1]]) *# If we are at the last step of a cycle, print a space character. This is only to make the output easier to see, and the cycles easy to count. This does not affect the circular list in any way.
        if* j%size(array)[1] == (size(array)[1]-1)
            print(“ ”)
 *end
    end
end**# Test the function on our custom 0-based array and also on our 1-based array for comparison*print_circular(a, 5)
println(“\n”)
print_circular(b, 5)*# OUTPUT:
# The 0-based array works fine, while the 1-based array throws an error.*
```

# 最后一点

同样，正如代码中提到的，我们没有改变 Julia 的行为。我们仍然可以用正常的方式创建数组，它们将从索引 1 开始。当一个函数返回一个元组或数组时，它的索引仍然从 1 开始。我们只创建了一个自定义对象，它的工作方式完全像一个数组，并且有自定义索引，而不是传统的基于 1 的索引。