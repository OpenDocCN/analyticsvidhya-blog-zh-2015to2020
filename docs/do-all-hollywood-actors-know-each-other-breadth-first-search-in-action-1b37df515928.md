# 好莱坞的演员都互相认识吗？“行动”中的广度优先搜索

> 原文：<https://medium.com/analytics-vidhya/do-all-hollywood-actors-know-each-other-breadth-first-search-in-action-1b37df515928?source=collection_archive---------19----------------------->

![](img/c016b3de4ce390b4b044091b57956f3e.png)

照片由 [Vincentas Liskauskas](https://unsplash.com/@vincentas_) 在 [Unsplash](https://unsplash.com/photos/ErMkvcFla74) 拍摄

任务很简单:给定两个演员的名字，通过他们的电影角色找到他们之间的联系。比如艾玛·沃森通过丹尼尔·雷德克里夫(哈利·波特)认识了詹妮弗·劳伦斯，认识了詹姆斯·麦卡沃伊(维克多·弗兰肯斯坦)，最终认识了詹妮弗·劳伦斯(黑凤凰)。 [GitHub](https://github.com/dtemir/harvard-CS50AI) 。

```
$ python degrees.py large
Loading data...
Data loaded.
Name: Emma Watson
Name: Jennifer Lawrence
3 degrees of separation.
1: Emma Watson and Daniel Radcliffe starred in Harry Potter and the Chamber of Secrets
2: Daniel Radcliffe and James McAvoy starred in Victor Frankenstein
3: James McAvoy and Jennifer Lawrence starred in Dark Phoenix
```

这个任务是关于解决一个搜索问题，比如一个迷宫，其中我们被给定一个**初始状态**(艾玛·沃森)和一个**目标状态**(詹妮弗·劳伦斯)。我们需要找到初始状态和目标状态之间的 ***最短*** 路径，这意味着我们应该利用基于使用**队列边界**的**广度优先搜索**。通过使用队列边界，我们可以从初始状态开始逐渐检查所有可用的**相邻节点**，这意味着我们将总是到达最短路径，因为我们将穷尽所有状态，直到我们到达最短解。

首先，我们需要了解我们可以获得哪些数据。数据由 IMDb 提供，包括三个表格: *movies.csv* ，带有*片名、年份、*和*id 列表；* *stars.csv* 带列表的*person _ id*和他们的*movie _ id；*和 *people.csv* 与*id、姓名、*和*出生年份*的列表。

其次，我们需要确定**节点**将存储什么。在我们的例子中，节点类是一个具有`state`、`parent`和`action`属性的对象。

```
class Node():
    def __init__(self, state, parent, action):
        self.state = state    # the current actors id
        self.parent = parent  # the previous actors id
        self.action = action  # the current movie id
```

一旦我们**到达最终状态**并且**需要跟踪我们所做的路径**，存储`parent`将会非常方便。把它看作一个**单链表**，指针指向下一个节点(尽管在我们的例子中，它指向前一个节点)。

所以当谈到**广度优先搜索**时，我们应该使用一个**队列**来存储我们需要访问的所有节点。使用队列对于 BFS 来说是必不可少的，因为我们需要依次遍历同一层上的所有节点来找到最短路径。有一篇关于 BFS 的不错的媒体文章。

说到这里，现在我们可以开始我们的`shortest_path` 函数了，它接受 source_id 和 target_id 作为属性。我们首先初始化`frontier`，然后添加存储状态的`source_id`和父动作的`None`的第一个节点。因此，保持`explored`设定以避免检查相同的人和他们的电影是很重要的。

```
frontier = QueueFrontier()
frontier.add(Node(state=source_id, parent=None, action=None))
explored = set()
```

然后，我们遍历，直到我们找到解决方案或用尽所有可能的边界节点。如果边界是空的，我们可以说没有联系。如果**不为空**，我们从队列中提取节点，特别是我们先放在那里的那个。**如果节点的状态等于** `**target_id**`，说明我们已经到了解，需要确定我们做的路径。这就是节点的父节点派上用场的地方(耶！).我们使用节点进行遍历，并保存我们在`path`中找到的所有节点。一旦我们把所有的节点放在列表中，我们反转它，并返回它。

```
while True:
    if frontier.empty():  # No more available nodes to check against target_id
        raise Exception("No solution")
    node = frontier.remove()  # Take out the first element from the queue
    if node.state == target_id:  # We found the final state!
        path = []
        while node.parent is not None:
            path.append((node.action, node.state))
            node = node.parent
        # Don't forget to reverse it because we were moving from the final state to the initial
        path.reverse()
        return path
```

这还没有结束。请原谅我。**如果节点的状态不等于** `**target_id**`，那么我们将它添加到探索集，并将它的所有邻居添加到边界。

```
explored.add((node.action, node.state))  # Record that we've already seen this actor and movie
for action, state in neighbors_for_person(node.state):
    if not frontier.contains_state(state) and (action, state) not in explored:
        # New node for the next actor and movie
        child = Node(state=state, parent=node, action=action)
        frontier.add(child)  # By adding to the frontier, we put it in the end of the queue
```

你可能想知道`neighbors_for_person` 函数在做什么？嗯，它只是返回所有主演过同一部电影的邻近演员。例如，假设 Emma Watson 是初始状态，我们提取她所有电影(可能是所有哈利波特电影)的 id，然后检查它们以及参与这些电影的人。我们最终返回所有这些人的 id 和电影 id 的集合。

```
movie_ids = people[person_id]["movies"]
neighbors = set()
for movie_id in movie_ids:
    for person_id in movies[movie_id]["stars"]:
        neighbors.add((movie_id, person_id))
return neighbors
```

一般来说，这个算法肯定是相当慢的。`shortest_path`函数的运行时复杂度约为 O(n⁴，这非常糟糕，但该项目的重点是在真实世界的数据中应用 BFS。我希望你喜欢阅读这篇文章。请注意，这个项目是我哈佛 CS50 人工智能课程的一部分；你可以在这里找到更多关于它的信息。也请查看我的 [GitHub](https://github.com/dtemir/harvard-CS50AI) 库。

另一个例子:

```
python degrees.py large
Loading data...
Data loaded.
Name: Eddie Murphy
Name: Bill Clinton
3 degrees of separation.
1: Eddie Murphy and Halle Berry starred in Boomerang
2: Halle Berry and Robert Downey Jr. starred in Gothika
3: Robert Downey Jr. and Bill Clinton starred in The Last Party
```