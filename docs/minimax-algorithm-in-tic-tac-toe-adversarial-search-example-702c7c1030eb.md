# 井字游戏中的极大极小算法:对抗性搜索示例

> 原文：<https://medium.com/analytics-vidhya/minimax-algorithm-in-tic-tac-toe-adversarial-search-example-702c7c1030eb?source=collection_archive---------6----------------------->

![](img/f59b96e3191d46ec9a25c261fcc3f994.png)

照片由[**Kvistholt**](https://unsplash.com/@freeche)**[**Unsplash**](https://unsplash.com/photos/oZPwn40zCK4)**

**我那时大约八岁。我们会和比我大五岁的叔叔玩井字游戏，每次他不是赢就是平。我不怪他伤了我的自尊，但我在那个年纪一定是挺傻的。**

**今天，经过多年的学习和一年半的大学生活，我终于可以说，我实现了一个算法，可以玩井字游戏，从来没有输过。它被称为 **Minimax 决策规则**，这是一种**对抗性搜索**的类型，意味着这个算法面对的是一个正在与机器对抗的对手。例如，在井字游戏中，人工智能与人类对弈；AI 还**知道游戏的完整状态**(它可以看到整个情况，不像扑克的游戏)，这是 Minimax 算法的一个要求。**

**![](img/39a2e44a9c688e3b53c9633479885942.png)**

**来自我的 GitHub 的演示[回购](https://github.com/dtemir/harvard-CS50AI)**

**Minimax 使用每个状态的表示，将一个**获胜条件**标记为 1，将**失败情况**标记为-1，将**中立条件**标记为 0。因此，进入细节可能会让它听起来很复杂(至少对我来说是这样)，但让我们想象一下你和自己玩井字游戏**的情况。你可能想要**最大化**你的分数**最小化**你的组件分数(也是你，哈哈)，因此得名 **Minimax** 。当你为另一方比赛时，选择最大化你的分数和最小化你对手的分数的动作。你最终可能会做的是，你可能会开始**考虑你可以做的每一个动作**和你的对手可能做的**动作**。计算每一个可能的结果是 Minimax 的基石，因为这有助于人工智能选择最佳行动，这对对手来说也是最糟糕的。****

**好吧，如果这对你来说还不够专业，让我们更深入地研究一下。我们需要为 Minimax 算法实现的函数是`player`来确定哪个玩家当前正在采取行动(X 或 O)；`actions`查询哪些动作还可用(自由单元格)；`result`用提议的动作构建一个**假设**板，这意味着它是一个完整的副本，我们可以在上面放置我们的潜在动作(它必须是一个副本，以便在不修改原始动作的情况下探索所有可用状态)；`winner`确定游戏是否结束，谁是赢家；`terminal`看比赛是平局还是有人赢了；**重要**、**、**、`utility`计算**假想**棋盘的一个动作是导致**赢**还是**输**(minimax 的面包和黄油)；最后的`minimax`就是拿一块板，应用上面提到的所有函数，并返回**最优移动**。**

**所以，让我们回顾一下这些函数的实现。听起来工作量很大，但是大部分花费**不超过 10 行代码**。给定一个棋盘(这是一个简单的 2D 列表 3x3)，函数`player`告诉我们**该轮到谁了**。我们这里的目标是计算 x 和 o 的数量，找出哪个比另一个小，这意味着该轮到它们采取行动了。**

```
def player(board):
    Xs = 0
    Os = 0
    # simply iterate over the given 2D array and calculate how many Xs and Os are there
    for y_axis in board: 
        for x_axis in y_axis:
            if x_axis == X:
                Xs += 1
            elif x_axis == O:
                Os += 1
    # if numer of Xs is smaller or equal to Os, it is a turn for an X because it always goes first
    if Xs <= Os: 
        return X
    else:  # otherwise it is a turn for an O
        return O
```

**对于给定的电路板，函数`actions`告诉我们**可以采取什么行动**。我们在这里的目标是保存一组元组`(i, j)`，其中`i`是的行，`j`是表示空单元格的棋盘列。**

```
def actions(board):
    possible_actions = set() # set is used just to be sure there will only be unique tuples for y, y_axis in enumerate(board):
        for x, x_axis in enumerate(y_axis):
            # initial implementation puts variable EMPTY in all cells, which is equal to None
            if x_axis == EMPTY: 
                possible_actions.add((y, x)) return possible_actions
```

**给定一个棋盘和一个动作(这是我们应该填写的单元格的元组)，函数`result`返回棋盘的深度副本。什么是**深度复制**？这是一个完全相同的副本，但是是一个**独立的**对象，不与原始对象共享任何指针。我们需要一个深入的副本来探索我们可以采取什么行动来找到最大化或最小化分数的状态，而不修改原件。**

```
def result(board, action):
    if len(action) != 2:  # check if given action is a tuple of two elements
        raise Exception("result function: incorrect action")
    # check if given action is within the boundaries of the board (3x3)
    if action[0] < 0 or action[0] > 2 or action[1] < 0 or action[1] > 2:
        raise Exception("result function: incorrect action value") y, x = action[0], action[1]
    board_copy = copy.deepcopy(board) # using the imported library 'copy' # check if action is already there (even though we will call 'actions' before it)
    if board_copy[y][x] != EMPTY:
        raise Exception("suggested action has already been taken")
    else:  # here we use the player function to know which letter to put in the copy
        board_copy[y][x] = player(board) return board_copy
```

**`winner`函数，给定一个棋盘，**告知是否有赢家**。只在你和我之间，试图弄清楚如何写这个是一个忙乱的经历(当我想起它是一个 3x3 的网格时，它变得更好了)。**

```
def winner(board):
    # Since the board is always 3x3, I believe this approach is reasonable
    for y in range(3):
        # Check horizontal lines
        if (board[y][0] == board[y][1] == board[y][2]) and (board[y][0] != EMPTY):
            return board[y][0]
        # check vertical lines
        if (board[0][y] == board[1][y] == board[2][y]) and (board[0][y] != EMPTY):
            return board[0][y] # Check diagonals
    if (board[0][0] == board[1][1] == board[2][2]) or (board[0][2] == board[1][1] == board[2][0]) \
            and board[1][1] != EMPTY:
        return board[1][1] return None
```

**给定一个棋盘，函数`terminal`告知游戏是否结束。真的。就是这样，lol。**重要**，它很有用，因为我们想知道我们游戏的预期状态是否已经结束，这就是为什么它是一个独立的功能。**

```
def terminal(board):
    if winner(board) == X or winner(board) == O: # check if there is a winner
        return True
    # check if there is a tie (if no cells left and neither X nor O won)
    elif EMPTY not in board[0] and EMPTY not in board[1] and EMPTY not in board[2]:
        return True
    else: # otherwise return that the game is still going on
        return False
```

**`utility`函数，给定一个棋盘，**通过返回 1、-1 或 0 来告诉 X 或 O** **是否赢了**。**重要的**，它在选择 Minimax 中的最优选择时很有用，因为它告诉我们给定的状态是赢还是输。**

```
def utility(board):
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0
```

**最后，`minimax`函数，给定一个棋盘，**返回当前玩家的最佳行动**。该算法首先检查棋盘是否是`terminal`，这意味着游戏已经结束，不能采取任何行动。如果没有，它接着检查该轮到谁采取行动。**重要**，如果 AI 玩 **X** ，它会尝试从所有最大值中选出最好的最小值；同样，如果人工智能为**或**比赛，它会尝试从最小值中选出最差的最大值。**

```
def minimax(board):
    if terminal(board):
        return None if player(board) == X:
        score = -math.inf
        action_to_take = None for action in actions(board):
            min_val = minvalue(result(board, action)) if min_val > score:
                score = min_val
                action_to_take = action return action_to_take elif player(board) == O:
        score = math.inf
        action_to_take = None for action in actions(board):
            max_val = maxvalue(result(board, action)) if max_val < score:
                score = max_val
                action_to_take = action return action_to_takedef minvalue(board):
    # if game over, return the utility of state
    if terminal(board):
        return utility(board)
    # iterate over the available actions and return the minimum out of all maximums
    max_value = math.inf  
    for action in actions(board):
        max_value = min(max_value, maxvalue(result(board, action))) return max_valuedef maxvalue(board):
    # if game over, return the utility of state
    if terminal(board):
        return utility(board)
    # iterate over the available actions and return the maximum out of all minimums
    min_val = -math.inf
    for action in actions(board):
        min_val = max(min_val, minvalue(result(board, action))) return min_val
```

**很难理解`minimax`函数，因为它涉及到调用一个函数(例如 maxvalue ),然后调用另一个函数(例如 minvalue ),后者调用第一个函数，这比递归(双递归？相互递归？).如果您对这些功能有任何疑问，请告诉我。**

**复杂性肯定不是该项目的最大优势，因为它涉及计算 3x3 板上的每个状态。你可以在为 O 比赛时感觉到这一点，因为 AI 在采取任何行动之前必须计算所有可能的状态。请注意，这个项目是我哈佛 CS50 人工智能课程的一部分；你可以在这里找到更多关于它的信息。我的 GitHub 报告和课程演示在[这里](https://github.com/dtemir/harvard-CS50AI)。**