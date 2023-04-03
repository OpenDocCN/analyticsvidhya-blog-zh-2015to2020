# 危险证词(GPL 2018)

> 原文：<https://medium.com/analytics-vidhya/dangerous-depositions-gpl-2018-f11588b0d13f?source=collection_archive---------20----------------------->

![](img/59c6b451e9c8d412074e6eb7574986ef.png)

[摇滚猴子](https://unsplash.com/@rocknrollmonkey?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

平衡机器人布雷特正在执行废黜其朋友的人类霸主的任务。在 N 个建筑物的城市中，他们计划 Q 次攻击，在那里他们偷偷穿过连接建筑物 A_i 和 B_i 与危险 D_i 的 M 条电缆。他们将从开始的建筑物 S_i 开始，行进到结束的建筑物 E_i。然后他们将潜入目的地建筑物，并以进步的名义将居住在该建筑物中的人类霸主扔出窗外。

路径的危险被定义为沿着该路径的任何电缆的最高危险值。布雷特希望最小化这种相关的危险，所以对于每次攻击，他们希望你，他们唯一信任的人，通过最小化他遇到的最大危险来计划每条路线，并找到他们在每条 Q 路线上将遇到的最大危险。最后，为了让他们放心，在承担这项危险任务之前，他们希望您输出所有电缆的危险等级总和，您可以承诺不会让他们在任何可能的未来攻击中使用这些电缆。

若要查看或尝试完整问题， [**在此报名**](https://www.hackerrank.com/contests/bennetts-problems) 然后[**点击此处**](https://www.hackerrank.com/contests/bennetts-problems/challenges/dangerous-depositions) **。**

# 解决办法

我们可以从简化问题陈述开始。我们被要求找到一系列危险最小的路径，给出一系列起点和终点。此外，我们需要输出避免的总危险的总和，即永远不会被遍历的边的总和。

要解决第一个问题，怎样才能找到危险度最低的路径？注意，给定从 A 到 B 的最大危险 X 和从 B 到 C 的最大危险 Y，从 A 到 C 的路径的最大危险≤ max(X，Y)。这是因为从 A 到 C 的路径可以遵循与 A 到 B，然后 B 到 C 相同的路径，并且消除了一些重复的边。如果我们能够创建一个具有最小最大加权边的连通子图，那么我们就可以在该子图中路由所有路径。我们可以通过贪婪地选择连接两个先前不连接的部分的最不危险的边来建立这个子图，继续直到子图被连接。然后，我们可以使用简单的深度优先搜索算法来找到一条危险最小的路径。我们稍后再讨论这个。

为了解决第二个问题，我们不可避免地必须识别哪些危险以及哪些边缘是永远不会穿越的。因此，我们也将知道哪些边被遍历。我们的解决方案必须最大化未遍历边的总和，最小化遍历边的总和。为了覆盖所有可能的路径，我们知道表示我们遍历的边的图必须是连通的。因此，我们需要找到一个最小权重的图来遍历代表问题的更大的图。一个大小为 N 的图需要连接 N-1 条边，任何有 N 条或更多条边的图都必然有回路。因此，我们知道这个最小权重图必须有 N-1 条边，否则我们可以删除一个回路中的一条边来创建一个较低权重的图。因此，后半部分只是要求我们找到一个最小生成树(MST)和不在该 MST 中的权重之和。

有趣的是，前半部分的解实际上会产生一个 MST。虽然没有被设定为 MST，但用于求解它的算法与用于寻找 MST 的 [Kruskal 的算法](https://en.wikipedia.org/wiki/Kruskal's_algorithm)相同。尽管提出了两个问题，但解决这两个问题只需要进行一个主要的计算，即寻找一个 MST。虽然 Kruskal 解决了这个问题，但实现起来可能有点麻烦，因为确定图的哪些部分已经连接起来非常耗时。知道我们的最终目标是找到一个 MST，我们可以使用 [Prim 的算法](https://en.wikipedia.org/wiki/Prim%27s_algorithm)，它通过选择“进行中”图与非连接点之间最便宜的链接，并将这条边添加到图中，直到生成树形成，从而逐步构建 MST。

一旦找到这个 MST，您只需要找到被请求路径的最小危险，并输出未使用的总权重。

下面是我用 C++的解决方案。

```
// Solution by Bennett Liu
#include <iostream>
#include <vector>
#include <queue>
#include <cstdio>
using namespace std;int n, m, q, a, b, d, s, e, avoided;
bool vis[1002];
pair<int, pair<int, int> > tmp;
vector<pair<int, pair<int, int> > > edges[1002];
vector<pair<int, int> > mst[1002];
priority_queue<pair<int, pair<int, int> > > pq;
int danger[1002][1002];void dfs(int cur, int origin, int curDanger) {
   danger[cur][origin] = curDanger;
   danger[origin][cur] = curDanger;
   vis[cur] = true;
   for (int i = 0; i < mst[cur].size(); i++) {
      if (!vis[mst[cur][i].second]) {
         dfs(mst[cur][i].second, origin, max(curDanger, mst[cur][i].first));
      }
   }
   return;
}int main() {
   // Get inputs
   cin >> n >> m >> q;
   for (int i = 0; i < m; i++) {
      cin >> a >> b >> d;
      edges[a].push_back(make_pair(-d, make_pair(a, b)));
      edges[b].push_back(make_pair(-d, make_pair(b, a)));
   }// Build MST and calculate sum of avoided dangers
   vis[1] = true;
   for (int i = 0; i < edges[1].size(); i++) pq.push(edges[1][i]);
   while (!pq.empty()) {
      tmp = pq.top();
      pq.pop();
      if (vis[tmp.second.second]) avoided += (-tmp.first);
      else {
         avoided -= (-tmp.first);
         vis[tmp.second.second] = true;
         mst[tmp.second.first].push_back(make_pair((-tmp.first), tmp.second.second));
         mst[tmp.second.second].push_back(make_pair((-tmp.first), tmp.second.first));
         for (int i = 0; i < edges[tmp.second.second].size(); i++) pq.push(edges[tmp.second.second][i]);
      }
   }// Precompute  all dangers
   for (int i = 1; i <= n; i++) {
      // Reset vis array
      for (int j = 1; j <= n; j++) vis[j] = false;
      dfs(i, i, 0);
   }// Return precomputed dangers
   for (int i = 0; i < q; i++) {
      cin >> s >> e;
      cout << danger[s][e] << endl;
   }
   cout << (avoided/2) << endl;
}
```

我为 2018 年女生编程联盟大赛写了这个问题，目标是找到一个 MST 算法的非常规用途。