# 定义我的素数方法并比较它们

> 原文：<https://medium.com/analytics-vidhya/defining-my-prime-number-methods-and-compare-them-6777047325bc?source=collection_archive---------18----------------------->

![](img/68a65fb63d8a6e4c664b328af56ccb86.png)

质数是一个大于 1 的整数，它的唯一因子是 1 和它本身。当我试图完成一个检查一个数是否是质数的方法时，很容易想出尝试用 2 到(n-1)范围内的数除数 n 的解法。所以基本思路是在 2 到(n-1)内，寻找 1 和数本身以外的因子。是的，这是我能想到的处理这个问题的第一个方法。以下是我的代码:

```
def prime0?(number)
  if number<2
     false
  else
      i=2
      while i < number
        if number % i == 0
          return false
        end
      i=i+1  
      end
   true
 end
```

是的，它工作得很好！但是后来我看代码的时候，真的觉得不够高效。然后，我想到了下面的想法，如果一个数肯定不是质数，换句话说，不是合数，那么其中一个因子会落在[2，n/2]中，因为因子总是成对出现，一个比 n/2 小，另一个比 n/2 大。所以我只需要在[2，n/2]的范围内检查，这减少了我的代码需要做的一半工作。下面是我如何实现我的第二个想法:

```
def prime1?(number) if number < 2
     false
  else
     i = 2
     while i < = number/2
       if number % i == 0
         return false
       end
     i=i+1
     end
    true
 endend
```

然而，即使它有效，我仍然试图找到一个更好的解决问题的方法。就在那时，我有了一个想法:如果我把计数器 I 设为 2，然后每次递增 1，会不会浪费很多时间？想想看，如果数字 n 能被 4 整除，那么它肯定能被 2 整除，对吗？也就是说，如果这个数从一开始就不能被 2 整除，那么我们可以把 I 设置为 3，然后递增 2，这样就省去了检查偶数的麻烦。这是我的代码:

```
def prime2?(number)
  if number < 2||number == 4
    false
  elsif number==2
    true
  else
   i=3
   while i <= number/2
     if number % i== 0
       return false
     end
   i=i+2
   end
   trueend
end
```

但是，我不确定你是否注意到了，这次我做的不仅仅是把 I 改成代码。因为我的“while”循环的条件是“i<=number/2 ”,所以对于数字 4，它的一半是 2，小于 I 最初设置的值，在数字 4 到达“if”语句后，它将跳过它并跳到“while”循环的结尾，这是真的。显然，这不是真的。这就是为什么我必须在第一个‘if’语句中硬编码它:

```
if number<2||number==4
   false
```

好，我测试了一下，这个方法也没问题。这是我如何测试我对质数 2 的方法比质数 1 快，质数 1 比质数 0 快。在阅读了 Jesse Storimer 的这篇文章后，我使用了基准#bm 方法来帮助我:

所以基本上，我简单地定义了 3 种方法来检查[2，10000]内的质数，然后使用#bm 方法并排生成报告，向我展示每种方法的速度，从而更容易比较哪种方法更有效。以下是我的代码:

```
def checkprime0
  i=2
  while i<10000
    prime0?(i)
    i=i+1
  end
end def checkprime1
  i=2
  while i<10000
    prime1?(i)
    i=i+1
  end
end def checkprime2
  i=2
  while i<10000
    prime2?(i)
    i=i+1
  end
end
  Benchmark.bm do |bm|
  bm.report { checkprime0}
  bm.report { checkprime1}
  bm.report { checkprime2}end
```

以下是结果:

```
user     system      total        real
0.244000   0.004000   0.248000 (  0.246886)
0.164000   0.000000   0.164000 (  0.163082)
0.144000   0.000000   0.144000 (  0.145867)
```

而如果我试着让它看起来更不一样，我可以试着把范围扩大到[2，100000]，结果是:

```
user     system      total        real
19.628000   0.000000  19.628000 ( 19.635854)
12.876000   0.000000  12.876000 ( 12.875490)
11.240000   0.000000  11.240000 ( 11.241820)
```

是的，结果，彼此相比，现在有了更大的差异！所以至少现在我确信我对哪一个会更快做出了正确的猜测(即使第二个和第三个没有太大的区别)。我很高兴我已经尝试过了，那很有趣！

所以最后，感谢你阅读我的博客。我仍然是编码新手，所以我上面写的并不完美。所以，如果你有更多关于如何更快的查质数的建议(我相信你会的！或者如果你认为这个博客有什么错误，请发邮件给我，我可以在这个博客中纠正它！谢谢！