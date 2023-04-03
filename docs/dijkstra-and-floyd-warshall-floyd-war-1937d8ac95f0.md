# 迪克斯特拉和弗洛伊德·沃肖尔

> 原文：<https://medium.com/analytics-vidhya/dijkstra-and-floyd-warshall-floyd-war-1937d8ac95f0?source=collection_archive---------17----------------------->

![](img/1a717a483aafdc529c03d9f28c1c7634.png)

托马斯·金托在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 5321.在阈值距离内找到邻居数量最少的城市

[https://leet code . com/problems/find-the-city-with-small-number of-neighbors-at-threshold-distance/](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/)

[Floyd Warshall](https://leetcode.com/problems/find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance/discuss/490275/Python-DP-based-on-Floyd-Warshall) 算法可以用来解决这个问题，因为问题规模相对较小。

```
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        dis = [[float('inf')]*n for _ in range(n)]
        for a,b, c in edges:
            dis[a][b]=dis[b][a]=c
        for i in range(n):dis[i][i]=0
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    dis[i][j] = min(dis[i][j], dis[i][k]+dis[k][j])
        res = {sum(d<=distanceThreshold for d in dis[i]):i for i in range(n)}
        return res[min(res)]
```

我们也可以用朴素的 Dijkstra 算法来解决这个问题。

```
import heapq
class Solution:
    def findTheCity(self, n: int, edges: List[List[int]], distanceThreshold: int) -> int:
        cost = [[float('inf')]*n for _ in range(n)]
        graph = collections.defaultdict(list)
        for a,b,c in edges:
            cost[a][b]=cost[b][a]=c
            graph[a].append(b)
            graph[b].append(a)

        def dijkstra(i):
            dis = [float('inf')]*n
            dis[i], pq =0, [(0, i)]
            heapq.heapify(pq)
            while pq:
                # the extract min which is heappop here has the complexit of O(logv) so the whole complexity is O(VlogV)d, i = heapq.heappop(pq)
                if d> dis[i]:continue
                for j in graph[i]:
                    this_cost = d+cost[i][j]
                    if this_cost<dis[j]:
                        dis[j] = this_cost
# the push operation if we use Fibonacci heap, the complexity is O(1), so the complexity is O(E). Hence the overrall complexity is O(E+VlogV) heapq.heappush(pq, (this_cost, j))
            return sum(d<=distanceThreshold for d in dis)-1
        res = {dijkstra(i):i for i in range(n)}
        return res[min(res)]
```

为了全面理解 dijkstra 算法，请阅读维基百科或下面的博客(中文)。

[](https://blog.csdn.net/v_JULY_v/article/details/6182419) [## 经典算法研究系列：二之续、彻底理解 Dijkstra 算法 _ 网络 _ 结构之法 算法之道-CSDN 博客

### 经典算法研究系列：二之续、彻底理解 Dijkstra 算法 作者：July 二零一一年二月十三日。参考代码：introduction to algorithms，Second…

blog.csdn.net](https://blog.csdn.net/v_JULY_v/article/details/6182419) 

看起来 C++是在用二进制堆而不是 Fibonacci 堆来实现优先级队列。讨论可以在这里找到。在这种情况下，Dijkstra 算法的时间复杂度将为 O(VlogV+ElogV)= O(V+E)logV = O(ElogV)。当 V 远小于 E 时，斐波那契堆将是更好的实现。

[https://www . quora . com/Why-is-the-c++-STL-priority-queue-implemented-using-a-binary-heap-代替-a-Fibonacci-heap](https://www.quora.com/Why-is-the-C++-STL-priority-queue-implemented-using-a-binary-heap-instead-of-a-Fibonacci-heap)

这里有一个关于 Dijkstra 算法的很好的解释

[](https://en.wikipedia.org/wiki/Fibonacci_heap) [## 斐波那契堆

### 在计算机科学中，斐波纳契堆是一种用于优先级队列操作的数据结构，由一组…

en.wikipedia.org](https://en.wikipedia.org/wiki/Fibonacci_heap) 

不同堆的复杂性可以从上面的链接中找到。这里有一个简短的总结:

二进制堆和 Fibonacci 堆的复杂度在 delete-min 上是 O(log V ),在 find-min 上是 O(1)。然而，Fibonacci 对于 insert 和 decrease 键都具有 O(1)的复杂度。而对于这两种操作，二进制堆的复杂度都是 O(log V)。

对于本帖提到的问题的代码实现，我们没有使用减键操作。相反，使用了本机插入，在这种情况下，我们可能在队列中有重复的节点，因为一些节点有更大的键。这将以某种方式破坏 O(VlogV)的复杂性，因为我们可能有 KV 而不是 V，其中 K 是指定节点的平均重复数。但是，这样做，Dijkstra 的实现就简单多了。我们将时间复杂度从 O(VlogV+E)增加到 O(KVlogKV+E ),将空间复杂度从 O(V)增加到 O(KV)。

似乎 C++和 JAVA 都不支持减键操作。而是使用惰性删除。

“虽然我的回答不会回答原来的问题，但我认为它可能对试图在 C++/Java 中实现 Dijkstra 算法的人(像我自己)有用，这是 OP 的一个评论，

C++中的`priority_queue`(或 Java 中的`PriorityQueue`)不提供`decrease-key`操作，如前所述。在实现 Dijkstra 时使用这些类的一个很好的技巧是使用“惰性删除”。Dijkstra 算法的主循环从优先级队列中提取下一个要处理的节点，并分析其所有相邻节点，最终改变优先级队列中节点的最小路径的成本。这就是通常需要`decrease-key`来更新该节点的值的地方。

诀窍是根本不要改变*它*。相反，该节点的“新副本”(具有其新的更好的成本)被添加到优先级队列中。由于成本较低，该节点的新副本将在队列中的原始副本之前被提取，因此它将被更早地处理。

这种“懒惰删除”的问题是，具有较高坏成本的节点的第二个副本最终将从优先级队列中提取出来。但是这总是发生在具有更好成本的第二个添加的副本被处理之后。因此*当从优先级队列中提取下一个节点时，主 Dijkstra 循环必须做的第一件事*是检查该节点是否以前被访问过(并且我们已经知道最短路径)。正是在这个时候，我们将进行“惰性删除”，该元素必须被简单地忽略。

这个解决方案在内存和时间上都是有代价的，因为优先级队列存储了我们没有删除的“死元素”。但是真正的成本将会非常小，而且，依我看，对这种解决方案进行编程，比试图模拟缺失的`decrease-key`操作的任何其他替代方案都要容易”

详情可从下面的链接中找到。

[](https://stackoverflow.com/questions/9209323/easiest-way-of-using-min-priority-queue-with-key-update-in-c/9210662) [## 在 C++中使用带有密钥更新的最小优先级队列的最简单方法

### 嗯，正如 Darren 已经说过的，std::priority_queue 没有降低元素优先级的方法，而且…

stackoverflow.com](https://stackoverflow.com/questions/9209323/easiest-way-of-using-min-priority-queue-with-key-update-in-c/9210662) 

## 使用标准 Dijkstra 算法的另一个问题是

## 1514 年。概率最大的路径

```
class Solution {
public:
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end) {
        // using this 2d vector will get MLE. 
       // vector<vector<float>>cost(n, vector<float>(n, 0));
        map<pair<int,int>, float> cost;
        unordered_map<int, vector<int>> graph;
        for(int i=0;i<edges.size();++i)
        {
            auto a = edges[i][0], b = edges[i][1];
            if(a<b)
            {
                cost[make_pair(a,b)] = succProb[i];
            }
            else
            {
                cost[make_pair(b,a)] = succProb[i];
            }
            graph[a].push_back(b);
            graph[b].push_back(a);
        }

        vector<float> dis(n, 0.0);
        dis[start] = 1.0;
        priority_queue<pair<float, int>> pq;
        pq.emplace(1.0, start);
        while (!pq.empty())
        {
            auto d = pq.top().first;
            auto i = pq.top().second;
            pq.pop();
            if (d<dis[i])continue;
            for (auto j: graph[i])
            {
                auto p = 0.0;
                if(i<j)
                {
                    if(cost.count(make_pair(i,j)))p=cost[make_pair(i,j)];
                }
                else
                {
                    if(cost.count(make_pair(j,i)))p=cost[make_pair(j,i)];
                }
                auto this_cost = d*p;
                if (this_cost>dis[j])
                {
                    dis[j] = this_cost;
                    pq.emplace(this_cost, j);
                }
            }
        }
        return dis[end];
    }
};
```

我们也可以用更简洁的方式来写

```
typedef pair<int, double> id;
typedef pair<double , int> di;class Solution {
public:
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end) {
        vector<vector<id>> graph(n);
        for (int i=0; i<edges.size();++i)
        {
            auto u = edges[i][0], v = edges[i][1];
            auto p = succProb[i];
            graph[u].emplace_back(v, p);
            graph[v].emplace_back(u, p);
        }
        vector<double> dis(n, 0.0);
        dis[start] = 1.0;
        priority_queue<di> Q;
        Q.emplace(1.0, start);
        while (!Q.empty())
        {
            auto [cost, u] = Q.top();
            Q.pop();
            // we can not set the condition here as cost<=dis[u], as the first time when we are going to further explore, the condition is exactly cost==dis[u]. for example, ititially the dis[1] = 0.0\. Then we updte it dis[1] = 0.5 and put it into the priority queue. Later, when we be back again, we did not explore the neighbors of 1, we SHOULD explore it. And the condition here is exactly dis[1] == 0.5.
            if (cost <dis[u]) continue;
            for (auto [v, p]: graph[u])
            {
                auto this_cost = cost*p;
                if(this_cost>dis[v])
                {
                    dis[v] = this_cost ;
                    Q.emplace(this_cost, v);
                }
            }
        }
        cout<<endl;
        return dis[end];
    }
};
```

Dijkstra 算法的标准实现

```
typedef pair<int, double> id;
typedef pair<double , int> di;class Solution {
public:
    double maxProbability(int n, vector<vector<int>>& edges, vector<double>& succProb, int start, int end) {
        vector<vector<id>> graph(n);
        for (int i=0; i<edges.size();++i)
        {
            auto u = edges[i][0], v = edges[i][1];
            auto p = succProb[i];
            graph[u].emplace_back(v, p);
            graph[v].emplace_back(u, p);
        }
        vector<double> dis(n, -1.0);
        priority_queue<di> Q;
        Q.emplace(1.0, start);
        while (!Q.empty())
        {
            auto [cost, u] = Q.top();
            Q.pop();
            if (dis[u] !=-1.0) continue;
            dis[u] = cost;
            if (u==end) return dis[end]==-1.0?0.0:dis[end]; 
            for (auto [v, p]: graph[u])
            {
                auto this_cost = cost*p;
                Q.emplace(this_cost, v);
            }
        }
        return  dis[end]==-1.0?0.0:dis[end];
    }
};
```

# 为什么在 Dijkstra 算法中不允许负权边？

参见[2]。

# 另一个例子

我们也可以使用 Floyd Warshall 算法来解决下面的问题。

[1462 年。课程表四](https://leetcode.com/problems/course-schedule-iv)

```
int dp[100+10][100+10];
class Solution {
public:
    vector<bool> checkIfPrerequisite(int n, vector<vector<int>>& prerequisites, vector<vector<int>>& queries) 
    {
        memset(dp, 0, sizeof(dp));
        for (auto p : prerequisites)
        {
            dp[p[0]][p[1]] = 1;
        }
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    dp[i][j] = dp[i][j] || (dp[i][k] && dp[k][j]);
        vector<bool> ret;
        for (auto q : queries)
            ret.push_back(dp[q[0]][q[1]]);
        return ret;
    }
};
```

# 参考

[1]https://youtu.be/DAaEMGJk70A

[2]https://blog.csdn.net/bhh77611355/article/details/27181285

[3][https://www . cnblogs . com/Gao chun dong/p/Dijkstra _ algorithm . html](https://www.cnblogs.com/gaochundong/p/dijkstra_algorithm.html)