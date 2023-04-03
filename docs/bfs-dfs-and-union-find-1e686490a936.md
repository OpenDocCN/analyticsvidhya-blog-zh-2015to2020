# BFS、DFS 和 Union Find

> 原文：<https://medium.com/analytics-vidhya/bfs-dfs-and-union-find-1e686490a936?source=collection_archive---------16----------------------->

我将通过一个例子展示 BFS(广度优先搜索)、DFS(深度优先搜索)和联合查找之间的联系。

例子是 leetcode 的一个问题。

问题是 1263。将盒子移动到目标位置的最少移动次数

[https://leet code . com/problems/minimum-moves-to-move-a-box-to-the-target-location/](https://leetcode.com/problems/minimum-moves-to-move-a-box-to-their-target-location/)

这是一个难题。难的原因是它含有两个 bfs。

这个问题的解决方案可以是:BFS + BFS 或者 BFS + DFS 或者 BFS + Union Find

解决方案 1(BFS + BFS，运行时间 160 毫秒)

解决方案 1 通过使用德克改进了 BFS+BFS(116 毫秒)

解决方案 2(BFS + DFS 运行时间 632 毫秒)

```
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 'S':
                    pi, pj = i, j
                elif grid[i][j] == 'B':
                    bi, bj = i, j

        def dfs_person(i, j, ti, tj, bi, bj):
            seen = set()
            if ti>=R or tj>=C or grid[ti][tj]=='#':return False
            open_list = [(i,j)]
            while open_list:
                i,j = open_list.pop()
                if (i,j)==(ti,tj):return True
                seen.add((i, j))
                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                    r, c = i+di, j+dj
                    if 0<=r<R and 0<=c<C and (r,c)!=(bi,bj) and (r,c) not in seen and grid[r][c]!='#':
                        open_list.append((r,c))
            return False

        def bfs(i, j, pi, pj):
            b_seen = set()
            cur_level = {(i,j, pi, pj, 0)}
            while cur_level:
                nxt_level = set()
                for i, j, pi, pj, d in cur_level:
                    b_seen.add((i,j, pi, pj))
                    if grid[i][j] == 'T':return d
                    for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                        r, c = i+di, j+dj
                        if 0<=r<R and 0<=c<C and grid[r][c]!='#' and (r,c, i, j) not in b_seen:
                            ti, tj = i-di, j-dj
                            if dfs_person(pi, pj, ti, tj, i, j):
                                nxt_level.add((r,c,i, j, d+1))
                cur_level = nxt_level      
            return -1
        return bfs(bi, bj, pi, pj)
```

解决方案 3(BFS +联合查找运行时间 1608 毫秒)

```
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        self.uf = {}
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 'S':
                    pi, pj = i, j
                elif grid[i][j] == 'B':
                    bi, bj = i, j
        def find(x):
            self.uf.setdefault(x, x)
            if self.uf[x] != x:
                self.uf[x] = find(self.uf[x])
            return self.uf[x]

        def union(x, y):
            self.uf[find(y)] = find(x)

        def union_find(i, j, bi, bj):
            open_list = [(i,j)]
            seen = set()
            while open_list:
                i,j = open_list.pop()
                seen.add((i, j))
                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                    r, c = i+di, j+dj
                    if 0<=r<R and 0<=c<C and (r,c)!=(bi,bj) and (r,c) not in seen and grid[r][c]!='#':
                        union((i, j), (r, c))
                        open_list.append((r, c))                   

        def bfs(i, j, pi, pj):
            b_seen = set()
            cur_level = {(i,j, pi, pj, 0)}
            while cur_level:
                nxt_level = set()
                for i, j, pi, pj, d in cur_level:
                    b_seen.add((i,j, pi, pj))
                    if grid[i][j] == 'T':return d
                    children = [(i+di, j+dj, i-di, j-dj) 
                                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]
                                if 0<=i+di<R and 0<=j+dj<C and grid[i+di][j+dj]!='#' and (i+di, j+dj, i, j) not in b_seen]
                    if children:
                        self.uf = {}
                        union_find(pi, pj , i, j)
                        for r, c, ti, tj in children:
                            if find((ti, tj)) == find((pi, pj)):
                                nxt_level.add((r,c,i, j, d+1))
                cur_level = nxt_level      
            return -1
        return bfs(bi, bj, pi, pj)
```

解决方案 4(BFS + DFS TLE)这个解决方案的原因可能是如果我们在整个列表中保留所有级别的元素，那么 open_list 太大了。方案二与此方案相比，只改变了 open_list 部分，只维持当前级别和下一级别。方案 2 的速度要快得多，可以被接受。

```
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 'S':
                    pi, pj = i, j
                elif grid[i][j] == 'B':
                    bi, bj = i, j

        def dfs_person(i, j, ti, tj, bi, bj):
            seen = set()
            if ti>=R or tj>=C or grid[ti][tj]=='#':return False
            open_list = [(i,j)]
            while open_list:
                i,j = open_list.pop()
                if (i,j)==(ti,tj):return True
                seen.add((i, j))
                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                    r, c = i+di, j+dj
                    if 0<=r<R and 0<=c<C and (r,c)!=(bi,bj) and (r,c) not in seen and grid[r][c]!='#':
                        open_list.append((r,c))
            return False                     

        def bfs(i, j, pi, pj):
            b_seen = set()
            open_list = [(i,j, pi, pj, 0)]
            for i, j, pi, pj, d in open_list:
                b_seen.add((i,j, pi, pj))
                if grid[i][j] == 'T':return d
                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                    r, c = i+di, j+dj
                    if 0<=r<R and 0<=c<C and grid[r][c]!='#' and (r,c, i, j) not in b_seen:
                        ti, tj = i-di, j-dj
                        if dfs_person(pi, pj, ti, tj, i, j):
                            open_list.append((r,c,i, j, d+1))    
            return -1
        return bfs(bi, bj, pi, pj)
```

实际上，解决方案 4 get TLE 并不是因为 open_list 太大。原因是太多的重复值被放入开放列表。当将代码更改为以下代码时，它可以工作，运行时间为 348 毫秒。

解决方案 5 (BFS + DFS 返回一个可见集 788 毫秒)在这种情况下，联合查找似乎比 DFS 慢。

```
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 'S':
                    pi, pj = i, j
                elif grid[i][j] == 'B':
                    bi, bj = i, j

        def dfs_person(i, j, bi, bj):
            seen = set()
            open_list = [(i,j)]
            while open_list:
                i,j = open_list.pop()
                seen.add((i, j))
                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                    r, c = i+di, j+dj
                    if 0<=r<R and 0<=c<C and (r,c)!=(bi,bj) and (r,c) not in seen and grid[r][c]!='#':
                        open_list.append((r,c))
            return seen                     

        def bfs(i, j, pi, pj):
            b_seen = set()
            cur_level = {(i,j, pi, pj, 0)}
            while cur_level:
                nxt_level = set()
                for i, j, pi, pj, d in cur_level:
                    b_seen.add((i,j, pi, pj))
                    if grid[i][j] == 'T':return d
                    children = [(i+di, j+dj, i-di, j-dj) 
                                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]
                                if 0<=i+di<R and 0<=j+dj<C and grid[i+di][j+dj]!='#' and (i+di, j+dj, i, j) not in b_seen]
                    if children:
                        seen = dfs_person(pi, pj , i, j)
                        for r, c, ti, tj in children:
                            if (pi, pj) in seen and (ti, tj ) in seen:
                                nxt_level.add((r,c,i, j, d+1))
                cur_level = nxt_level      
            return -1
        return bfs(bi, bj, pi, pj)
```

解决方案 6 (BFS + DFS 将 DFS 的 open_list 的数据类型从 list 改为 set runtime 192 ms)不知道为什么它能有如此显著的加速。列表和集合之间的追加、遍历和弹出速度应该不会太快。这部分还是不清楚。

```
class Solution:
    def minPushBox(self, grid: List[List[str]]) -> int:
        R, C = len(grid), len(grid[0])
        for i in range(R):
            for j in range(C):
                if grid[i][j] == 'S':
                    pi, pj = i, j
                elif grid[i][j] == 'B':
                    bi, bj = i, j

        def dfs_person(i, j, ti, tj, bi, bj):
            seen = set()
            if ti>=R or tj>=C or grid[ti][tj]=='#':return False
            open_list = {(i,j)}
            while open_list:
                i,j = open_list.pop()
                if (i,j)==(ti,tj):return True
                seen.add((i, j))
                for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                    r, c = i+di, j+dj
                    if 0<=r<R and 0<=c<C and (r,c)!=(bi,bj) and (r,c) not in seen and grid[r][c]!='#':
                        open_list.add((r,c))
            return False

        def bfs(i, j, pi, pj):
            b_seen = set()
            cur_level = {(i,j, pi, pj, 0)}
            while cur_level:
                nxt_level = set()
                for i, j, pi, pj, d in cur_level:
                    b_seen.add((i,j, pi, pj))
                    if grid[i][j] == 'T':return d
                    for di, dj in [(1, 0),(-1, 0), (0, 1), (0, -1)]:
                        r, c = i+di, j+dj
                        if 0<=r<R and 0<=c<C and grid[r][c]!='#' and (r,c, i, j) not in b_seen:
                            ti, tj = i-di, j-dj
                            if dfs_person(pi, pj, ti, tj, i, j):
                                nxt_level.add((r,c,i, j, d+1))
                cur_level = nxt_level      
            return -1
        return bfs(bi, bj, pi, pj)
```

通过这篇文章，希望读者能更好地理解 BFS、DFS 和 union-find 之间的区别和联系。