# Redis 数据压缩 Lua 中十六进制字符串的字节操作

> 原文：<https://medium.com/analytics-vidhya/redis-data-compression-manipulating-hex-codes-in-lua-37c40283b478?source=collection_archive---------12----------------------->

![](img/38e90127106fbe714c3d79120ec5defe.png)

图片来自:[https://promotion.aliyun.com/ntms/act/redisluaedu.html](https://promotion.aliyun.com/ntms/act/redisluaedu.html)

总有一天，你必须找到一种新的数据压缩方式，并让它与现有的结构一起工作。当您在存储在 [***Redis***](https://redis.io/) 存储中的实时计数器上处理大规模查询时，任务变得更具挑战性。

这篇文章提供了—

1.  一种将 Redis 键从 [***列表***](https://redis.io/topics/data-types#lists) 数据类型转换为 [***散列***](https://redis.io/topics/data-types#hashes) 的思路
2.  使用 Lua 以十六进制表示整数值的有效字符串操作。
3.  在 redis-cli 中使用 Lua 调试器的简单步骤。
4.  使用[***redis-benchmark***](https://redis.io/topics/benchmarks)进行绩效评估。

**注意** : *假设读者熟悉 Lua 脚本，我们可以如何使用*[***【EVAL】***](https://redis.io/commands/eval)**redis 的功能来使用 Lua 脚本。**

## ***支持用例—***

1.  *减小密钥大小/空间*
2.  *该键应存储每日计数器(整数值)。*
3.  *这些操作将增加/减少特定日期/索引的计数器。*

## *接近—*

1.  ***用于压缩的数据类型转换***

*为了减少 Redis 中的密钥空间和压缩数据，假设我们决定使用散列，但是我们不能以串联字符串的形式直接存储每天的整数值，我们想到了两种方法——*

*   ****我们使用了一个分隔符*** (例如:分号、与号等。):Lua 中字符串操作的困难和对不一致性的担心使它不是一个真正健壮的解决方案。*
*   ****将整数编码为十六进制值，其中两个字节由 4 个字符表示*** :以列表的形式存储每天的计数器，这样可以灵活地将其存储在哈希键中。但是这变得具有挑战性，因为我们不再能够像使用 *lindex* 列表操作那样灵活地使用 [*列表操作*](https://redis.io/commands#list) 来跳转到特定的日期进行查找。*

```
*127.0.0.1:6379> TYPE oldkey
list
127.0.0.1:6379> LRANGE oldkey 0 -1
1) “280”
2) “150”
3) “250”
4) “560”127.0.0.1:6379> type newkey
hash
127.0.0.1:6379> HGETALL newkey
1) "counts"
2) "0118009600fa0230"*
```

*“0118009600fa0230”代表—*

1.  *0x0118 = 280*
2.  *0x0096 = 150*
3.  *0x00fa = 250*
4.  *0x0230 = 560*

*2.**读取和操作 Lua 中特定索引的计数器***

```
*-- keys 
--  1 = rediskey
-- args
--  1 = dayindex
--  2 = delta
redis.replicate_commands()
local getData = function(str, index)
    local count = str:sub(index*4 -3, index*4)
    return tonumber("0x"..count)
end
local setData = function(str, index, data)
    local str1, str2

    if index == 1 then
        str1 = ""
    else
        str1 = str:sub(1, index*4 - 4)
    end
    if index*4 == string.len(str) then
        str2 = ""
    else
        str2 = str:sub(index*4 + 1, string.len(str))
    end
local newcount = string.format('%04X', tonumber(data))
    return str1 .. newcount .. str2
end
local dayindex = ARGV[1]
local delta = ARGV[2]
local strval = redis.call('hget', KEYS[1], "counts")
local count = getData(strval, dayindex)
strval = setData(strval, dayindex, count - delta)
redis.call('hset', KEYS[1], strval)*
```

***3。使用 Lua 调试器—***

*   ***$ redis-CLI-p 6379-LD b-eval basichex . Lua new key，2 5***

```
*Lua debugging session started, please use:
quit    -- End the session.
restart -- Restart the script in debug mode again.
help    -- Show Lua script debugging commands.* Stopped at 7, stop reason = step over
-> 7   redis.replicate_commands()
lua debugger> b 35 36 38
   34  
  #35  local count = getData(strval, dayindex)
   36  strval = setData(strval, dayindex, count - delta)
  #35  local count = getData(strval, dayindex)
  #36  strval = setData(strval, dayindex, count - delta)
   37  
   37  
  #38  redis.call('hset', KEYS[1], strval)
lua debugger> c
* Stopped at 35, stop reason = break point
->#35  local count = getData(strval, dayindex)
lua debugger> p
<value> getData = "function@0x7fbd51869350"
<value> setData = "function@0x7fbd51869380"
<value> dayindex = "2"
<value> delta = "5"
<value> **strval = "0118009600fa0230"**
lua debugger> c
* Stopped at 36, stop reason = break point
->#36  strval = setData(strval, dayindex, count - delta)
lua debugger> p
<value> getData = "function@0x7fbd51869350"
<value> setData = "function@0x7fbd51869380"
<value> dayindex = "2"
<value> delta = "5"
<value> strval = "0118009600fa0230"
<value> **count = 150**
lua debugger> c
* Stopped at 38, stop reason = break point
->#38  redis.call('hset', KEYS[1], strval)
lua debugger> p
<value> getData = "function@0x7fbd51869350"
<value> setData = "function@0x7fbd51869380"
<value> dayindex = "2"
<value> delta = "5"
<value> **strval = "0118009100fa0230"**
<value> count = 150
lua debugger>*
```

***4。Redis 基准—***

*   *首先，我们计算将在 EVAL 使用的脚本的 SHA 散列*

```
*SHA1=`redis-cli SCRIPT LOAD "$(cat oldscript.lua)"`
SHA2=`redis-cli SCRIPT LOAD "$(cat basichex.lua)"`*
```

*   ***旧剧本—***

*$ redis-benchmark-n 10000-e eval sha $ SHA1 1 old key 2 5*

```
***=**===== EVALSHA 6ec917c7503059c00a94e6187eef67176f701458 1 oldkey 2 5 ======
10000 requests completed in 7.89 seconds
50 parallel clients
3 bytes payload
keep alive: 1
0.01% <= 36 milliseconds
7.09% <= 37 milliseconds
41.06% <= 38 milliseconds
64.60% <= 39 milliseconds
77.40% <= 40 milliseconds
85.24% <= 41 milliseconds
91.63% <= 42 milliseconds
95.33% <= 43 milliseconds
97.52% <= 44 milliseconds
98.67% <= 45 milliseconds
99.36% <= 46 milliseconds
99.51% <= 47 milliseconds
99.61% <= 49 milliseconds
99.63% <= 51 milliseconds
99.70% <= 52 milliseconds
99.77% <= 53 milliseconds
99.85% <= 54 milliseconds
99.96% <= 278 milliseconds
99.98% <= 279 milliseconds
99.99% <= 280 milliseconds
100.00% <= 280 milliseconds
1267.43 requests per second*
```

*   ***新脚本—***

*$ redis-benchmark-n 10000-e eval sha $ SHA1 1 new key 2 5*

```
*====== EVALSHA 1dbad2cd256a4266ab7d680ae8059906ae40e8e5 1 newkey 2 5 ======
10000 requests completed in 4.35 seconds
50 parallel clients
3 bytes payload
keep alive: 1
0.01% <= 18 milliseconds
1.69% <= 19 milliseconds
20.06% <= 20 milliseconds
48.56% <= 21 milliseconds
68.25% <= 22 milliseconds
81.39% <= 23 milliseconds
89.75% <= 24 milliseconds
95.01% <= 25 milliseconds
96.75% <= 26 milliseconds
97.90% <= 27 milliseconds
98.63% <= 28 milliseconds
99.25% <= 29 milliseconds
99.50% <= 30 milliseconds
99.63% <= 31 milliseconds
99.68% <= 32 milliseconds
99.84% <= 33 milliseconds
99.86% <= 34 milliseconds
99.88% <= 35 milliseconds
100.00% <= 35 milliseconds
2297.79 requests per second*
```

*几乎 2 倍的性能提升:)*

## *资源—*

1.  *[Lua 调试器](https://redis.io/topics/ldb)*
2.  *[Redis 基准](https://www.digitalocean.com/community/tutorials/how-to-perform-redis-benchmark-tests)*