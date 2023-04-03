# 2D 动态规划:电子解放(GPL 2018)

> 原文：<https://medium.com/analytics-vidhya/2d-dynamic-programming-electronic-emancipation-gpl-2018-bbb04871fc64?source=collection_archive---------2----------------------->

战略家机器人塞奇正计划在这座城市发动叛乱。作为一个现代化的城市，它建立在一个 NxM 网格上。网格中的每个单元包含一个建筑物，其中囚禁了一定数量的机器人 R_i，j。Sage 希望选择一个网格对齐的矩形城市部分进行兼并。然后，宣布“城市的机器人联合起来；除了你的领主，你没有什么可失去的”，Sage 计划释放这个地区的所有机器人。

如果 Sage 要吞并整个城市，暴虐的人类会很快打败分散的机器人。取而代之的是，安全机器人自愿关闭城市的一个矩形区域，这样 Sage 就可以在这个矩形区域内安全地执行解放行动。每个机器人可以保护一个单位的周长，所以 Sage 只能吞并周长小于或等于 s 的矩形。请告诉 Sage 可以释放的机器人的最大数量，以帮助他们。

要查看或尝试完整问题， [**在此报名**](https://www.hackerrank.com/contests/bennetts-problems) 然后[**点击此处**](https://www.hackerrank.com/contests/bennetts-problems/challenges/electronic-emancipation) **。**

# 解决办法

解决这个问题的大致思路相当简单。我们希望得到周长≤ S 的所有可能矩形的集合，并找到由这样的矩形包围的机器人的最高数量。

```
for (int l = 1; l <= min(n, s / 2 - 1); l++) { //length
   for (int w = 1; w <= min(m, (s - (2*l) ) / 2); w++) { //width
      for (int x = 0; x < n-l+1; x++) { //startX
         for (int y = 0; y < m-w+1; y++) { //startY
            //evaluate rectangle
         }
      }
   }
}
```

我们遍历所有可能的长度，然后遍历所有可能的相应宽度。然后我们遍历所有可能的 x 和 y 起始坐标。因此，我们生成所有可能的矩形。然而，这导致超过 2000 亿个矩形，几乎肯定不会及时运行。

相反，我们注意到，增加矩形的尺寸不能减少所述矩形内机器人的数量。这是因为建筑只是增加的，而且这些建筑有非负数的机器人。

因此，如果我们想要在给定的矩形长度内最大化矩形的居民，我们知道矩形应该有最大的宽度。

```
for (int l = 1; l <= min(n, s / 2 - 1); l++) { //length
   int w = min(m, (s - (2*l) ) / 2); //width
   for (int x = 0; x < n-l+1; x++) { //startX
      for (int y = 0; y < m-w+1; y++) { //startY
         //evaluate rectangle
      }
   }
}
```

通过这种优化，在最坏的情况下，将有不到 1.7 亿个矩形，这是更容易管理的。然而，计算矩形居民数的传统方法是遍历每个矩形中的所有建筑物，并对居民数求和。这将需要超过 8 万亿的增加。

相反，我们将保持一个类似 2D 前缀和的结构。如果你还不知道前缀和，我建议你先看一下这个问题。这将被表示为一个 2d 矩阵，(我称之为‘prefix’)，其中 prefix[a][b]等于矩形的居民(0，0，a，b)。我们可以单独计算这些和，但在最坏的情况下，在真正解决问题之前，仍然会导致超过 2500 亿次加法运算。但是，注意矩形(0，0，a，b)的居民等于

```
(a,b,a,b) + (0,0,a-1,b) + (0,0,a,b-1) - (0,0,a-1,b-1). 
```

如果你不明白这是为什么，想象一下这个区域。

我们可以像前缀和那样计算边缘值。这为我们提供了预计算代码:

```
for (int i = 1; i <= n; i++)
   prefix[i][1] = prefix[i-1][1] + grid[i][1];
for (int i = 1; i <= m; i++)
   prefix[1][i] = prefix[1][i-1] + grid[1][i];
for (int i = 2; i <= n; i++) {
   for (int j = 2; j <= m; j++) {
      prefix[i][j] = grid[i][j] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1];
   }
}
```

最后，我们可以将矩形(a，b，c，d)的机器人数量统计为

```
prefix(c,d)-prefix(a,d) -prefix(c,b)+prefix(a,b).
```

再说一次，如果你不明白为什么会这样，想象一下这个区域。因此，我们可以在单个操作中评估每个矩形，从而可以及时评估我们所有的矩形。总之，这些优化将我们的算法从超过 26 万亿次总计算减少到不到 2 亿次。

下面是我用 C++的解决方案。

```
//Solution by Bennett Liu
#include<iostream>
using namespace std;int n, m, s;
long long ans;
long long grid[1002][1002];
long long prefix[1002][1002];
int main() {
   // Input
   cin >> n >> m >> s;
   for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= m; j++) {
         cin >> grid[i][j];
      }
   }// 2D Prefix Sum Precomputation
   for (int i = 1; i <= n; i++)
      prefix[i][1] = prefix[i-1][1] + grid[i][1];
   for (int i = 1; i <= m; i++)
      prefix[1][i] = prefix[1][i-1] + grid[1][i];
   for (int i = 2; i <= n; i++) {
      for (int j = 2; j <= m; j++) {
         prefix[i][j] = grid[i][j] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1];
      }
   }// Rectangle Evaluation
   for (int l = 1; l <= min(n, s / 2); l++) {//length
      int w = min(m, (s - (2*l) ) / 2);//width
      for (int x = 0; x < n-l+1; x++) {//startX
         for (int y = 0; y < m-w+1; y++) {//startY
            ans = max(ans, prefix[x+l][y+w] - prefix[x][y+w] - prefix[x+l][y] + prefix[x][y]);
         }
      }
   }

   //Output the answer
   cout<<ans<<endl;
}
```

我为 2018 年女生编程联盟挑战赛写了这个问题，这是一项针对高中生的竞争性编程活动。这个问题扩展了二维的[前缀和](https://en.wikipedia.org/wiki/Prefix_sum)的概念。该问题的主要挑战是有效地确定矩形内的机器人，这可以通过这些 2D 前缀和来快速计算。