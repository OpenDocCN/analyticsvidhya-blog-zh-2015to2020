# 了解 OpenAI 基线源代码，让它做自玩！第四部分

> 原文：<https://medium.com/analytics-vidhya/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-4-a54e075386bf?source=collection_archive---------13----------------------->

![](img/0c4e24ae4dd215d07c5cf52498b2ba0f.png)

在这里，我们将最终尝试实现 OpenAI 基线的自玩功能！要看第 1 部分、第 2 部分和第 3 部分，请在此处查看，在此处查看，在此处查看。我认为如果没有这个基础，这将有点难以理解。我将尝试在基线中实现实际的自我游戏功能，所以我希望这很有趣！但是，我会谈谈我自己的工作流程，我不习惯，所以如果你觉得有些部分难以理解，请告诉我！

要查看代码并跟进，请点击查看[！](https://github.com/isamu-isozaki/baseline-selfplay)

# 一个要求

我的环境的一个要求是双方需要能够同时做决定。在这种情况下，我希望模型在环境迈出一步之前更新两边或任意数量的边。

这样我们就不会在每一个时间步发送某一方面的观察和更新环境。这是我最初的想法，但我认为它行不通。

那么，我们从选择的一端运行每个操作，然后连接它们，最后运行 step 并更新环境，怎么样？例如，在 step 函数中，如果我们循环遍历所有的边，并根据它们观察到的情况采取行动，然后更新环境，我们可以有效地使它看起来像我们有边数的额外环境！

# 环境又是如何产生的？

到目前为止，我们看到 make_env 提供的功能总体上获得了我们创建的环境，只是用附加功能包装了它。因此，原始功能不会改变。因此，我们仍然可以访问环境的属性。

这解释了为什么要完成我们的目标，我们需要

1.  在 worker 函数中，它需要将各方的动作一次发送到远程/环境。或者说的更具体一点，是需要步骤阶段和实际步骤阶段的准备。这可以在工人函数中完成，也可以在实际环境中完成。我想我会在实际环境中做！
2.  观察空间是基于相同的环境，玩家人数相同(他们可以不同，因为你看不到一些玩家等)。因此，工人也需要以适当的方式发回观察结果！

因此，我的想法是改变它

1.  如果环境有一个属性“边”，它将表示环境拥有的边的数量，nenv 乘以这个数量。

我不确定如何做到这一点，因为首先在 SubprocVecEnv 中创建环境，然后继续这样会很麻烦，因为在创建环境的过程中，我们首先需要环境的数量。

我很确定这是可能的，但是我想到的每一个方法都变得很混乱。因此，我决定在 common_arg_parser 中添加一个名为“no_self_play”的参数，该参数最初设置为 False，如果用户将

```
--no_self_play
```

这将是真的，从而使一个普通的环境没有边。

为此，我使用了 make_vec_env 函数

```
if not env_kwargs.get("no_self_play", True):
    num_env *= env_kwargs.get("sides", 2)#The default number of sides is 2
```

其中 no_self_play 是一个参数。)提供。然而，我发现用户发送的参数不在 build env 的 env_kwargs 参数中。最初，env_kwargs 被设置为 None

```
env_kwargs = env_kwargs or {}
```

使它成为一个空字典。

现在，由于我希望用户的参数放在 env_kwargs 参数中，我们只需要将 build_env 中的参数从 arg_parse 转换成一个字典，然后将它作为一个参数提供！

将 arg_parse 参数转换成一个字典可以很容易地完成(感谢 [Raymond](https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary) ！)

```
args_dict = vars(args)
```

老实说这很酷！

然后，我通过调用 make_vec_env

```
env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations, env_kwargs=args_dict)
```

然后，我们成功的增加了边数的环境！

2.在子过程中，只有 nenvs//侧多个环境实际运行，并将每侧多个行为指标应用于每个环境。

这可能不是最优雅的解决方案，因为我是所有花哨的线程和多处理的新手，所以如果你有更好的解决方案，请告诉我！

在 __init__ 函数中，我写道

```
if(hasattr(env_fns[0](), 'sides')):
   self.sides = env_fns[0]().sides
else:
   self.sides = 1
```

那么，由于 nenvs//1 == nenvs，代码做的事情应该没有区别。我还在进行一些 bug 测试，所以这可能会改变！

无论如何，在 env_fns 上的数组 _split 之前

```
env_fns = np.array_split(env_fns, self.nremotes)
```

我做了

```
env_fns = env_fns[:nenvs//self.sides]
```

只获得第一个 nenvs//self.sides 许多环境，因为其余的将不会被使用！然后，我将断言和 self.n_remotes 更改为

```
assert nenvs//self.sides % in_series == 0, "Number of envs must be divisible by number of envs to run in series"self.nremotes = nenvs //self.sides // in_series
```

在这里，我没有改变 n_envs 变量的值的原因是，因为在实际的 learn 函数中，它在环境中训练算法(至少在 ppo2 中)，我看到

```
nenvs = env.num_envs
```

稍后用于计算批次！所以，我不能改变这个数字，因为那样的话，我就不能计算环境中的边数了！

无论如何，总的来说，这应该使得在众多的环境* self.sides，nenv 动作中，每个 self.sides 动作形成一个调用正确环境的块！

然后，我转到 step_async 并对其进行了更改，这样它就可以通过在远程 i//self.sides 上调用 action i 的操作来在每一侧执行 sides number of action！

```
def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for j in range(len(actions)):
            for action in actions[j]:
                self.remotes[j].send(('step', action))

        self.waiting = True
```

更新:对不起，它最初没有工作，因为 self.remotes 的索引最初是错误的。固定 2019/12/8

# 问题

然而，这里有一个小问题。环境没有办法知道它正在发送的动作对应于哪一方！可以从第一面也可以从第二面，总的来说，很混乱。

为了解决这个问题，我在我的环境中添加了一个名为 side 的属性，该属性在 __init__ 函数中初始设置为 0，如下所示

# 我的环境

```
self.side = 0
```

然后，我修改了我的环境，这样你就可以设置你想要的动作对应哪一面

```
def step(self, action):
    side = self.side
    self.action[side] = action
    self.finished_sides[side] = 1
    self.side += 1
    self.side %= self.sides
    if self.finished_side[self.finished_side == 0].shape[0] == 0:
        self.update_step()   
return None, None, None, None
```

其中 self.finished_side 最初全为 0，边的大小如下

```
self.finished_side = np.zeros(self.sides)
```

我确信

```
self.finished_side[self.finished_side == 0].shape[0] == 0
```

这不是最好的方法，我以后会想出更好的方法，但是现在，它很有效！

update_step 运行环境并返回观察结果、奖励、是否完成以及所有方面的空字典

```
def update_step(self):
    self.game_step()
    done = False
    if t>=self.terminate_turn or self.end():
        done = True
    self.t += 1
    self.finishe_sides[...] = 0
    dones = [done for _ in range(self.sides)]
    infos = [{} for _ in range(self.sides)]
    return self.obs, self.rewards, dones, infos
```

在这里，我这样做是为了让所有返回的东西，比如 self.obs，基本索引显示它是哪一方。所以比如边 0 的观察可以在 self.obs[0]找到！

接下来，在 step_wait 中，由于返回了一些 Nones，我们需要稍微修改一下代码，使数据分布在所有环境中，这样模型就能看到正确的观察结果，并且每个动作都对应于正确的方面和环境。

# 步骤 _ 等待

现在，虽然 step_wait 函数最初是从

```
results = [remote.recv() for remote in self.remotes]
```

我们需要稍微修改一下

```
results = [self.remotes[i//self.sides].recv() for i in range(len(self.remotes)*self.sides)]
```

这将从每个远程获得双方的结果数！

然后，这只是为了使稍后的重置函数变得平滑，但是我决定对结果数组进行排序，以便第一个 len(self.remotes)或 nenvs // self.sides 将是实际的观察值，而不是在每个 i % self.sides = self.sides-1 响应中都出现观察值(这是因为只有当所有的边都记录了它们的动作时，我才得到响应)

所以，在一定程度上去除了 in_series 效果的 _flatten_list 函数之后(至少我是这么理解的！)我做了

```
data = results.copy()[self.sides-1::self.sides]
results = np.asarray(results)
results[:len(self.remotes)] = data
```

现在，完成了，如果 self.sides 大于 1(意味着这是一个自我游戏环境)，我通过我的可怕命名的方法 tactic_game_fix_results 传递结果变量！

```
def tactic_game_fix_results(self, results):
        for i in range(len(results)-1, -1, -1):
            for j in range(len(results[i])):
                results[i][j] = results[i//self.sides][j][i % self.sides]
        return results
```

基本上，这里发生的是，我从后面迭代结果(其中都是 none ),然后用代理 I 的实际观察、奖励、完成和信息覆盖这些 none。我这样做是为了让代理看到 i//self.sides 环境的 side i % self.sides，我认为这是相当一致的！j 只是在观察、奖励、完成和信息之间循环。所以，len(results[i])这里是 4！

因此，总的来说，step_wait 函数最终变成了

```
def step_wait(self):
        self._assert_not_closed()
        #do recv on the same remote several times
        results = [self.remotes[i//self.sides].recv() for i in range(len(self.remotes)*self.sides)]
        results = _flatten_list(results)
        data = results.copy()[self.sides-1::self.sides]
        results = np.asarray(results)
        results[:len(self.remotes)] = data
        #push the observations to the first portion of the results array.
        if self.sides > 1:
            results = self.tactic_game_fix_results(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos
```

我没有把数据部分放在 if self.sides > 1 中，因为如果 self.sides = 1，应该没有变化！

# 重置

由于 reset 函数需要返回观察结果，但不需要任何操作(因为它是初始状态)，我可以做类似这样的事情

```
def reset(self):
  self.__init__(**self.kwargs)
  return self.obs
```

在我的环境和子环境中，我将 reset 改为

```
def reset(self):
        self._assert_not_closed()
        for i in range(len(self.remotes)):
            self.remotes[i].send(('reset', None))
        obs = [self.remotes[i].recv() for i, _ in enumerate(self.remotes)]
        obs = _flatten_list(obs)
        if self.sides > 1:
            obs += [[None] for _ in range(len(self.remotes)*(self.sides-1))]
            obs = self.tactic_game_fix_results(obs)     
            obs = zip(*obs)
            obs = obs.__next__()
        return _flatten_obs(obs)
```

坦白地说，我并不特别干净，我也不以从美学角度来看它为荣。但不管怎样，最主要的变化是

```
if self.sides > 1:
    obs += [[None] for _ in range(len(self.remotes)*(self.sides-1))]
    obs = self.tactic_game_fix_results(obs)     
    obs = zip(*obs)
    obs = obs.__next__()
```

第一部分填充观察值，这样所有模型都有足够的观察值。self.tactic_game_fix_results 也应该是不言自明的！接下来的几行会发生什么？嗯，基本上，我试着做一些事情

```
obs, rews, dones, infos = zip(*results)
```

为了得到观察结果。这里基本上发生的是，reset 的 obs 的大小从[num_agents，1，rest of dims]变为[num_agents，rest_of_dims]。

然而，在 step 函数中，返回的是 4 个值 j 是 4，但是对于 reset 的观察，出现了一个问题，因为只返回了 1 个值，observation，j 开始在观察的边上迭代，这破坏了一切，对我来说是一个调试噩梦。基本上，我所做的就是将 self.obs 放在我的环境中的一个列表中，如下所示！

```
def reset(self):
  self.__init__(**self.kwargs)
  return [self.obs]
```

现在，经过一点调试，我发现了一些错误的另一个来源。这是 worker 函数中的 step_env 函数。它主要做的是当一个步骤被请求时，它只是从 env 的 step 函数中获得观察结果和所有好的输出，并在这样的过程中为所有环境发送它！

```
if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
```

实际实现是

```
def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info
```

在这里，我最初有点困惑，但重要的是要注意，这是您的环境的重置函数，而不是您的 SubprocVecEnv 或任何包装器的重置函数或步骤函数！

总之，在这里，我有一个问题。正是因为我把 ob 放在一个类似[ob]的列表中，并在 reset 函数中返回，所以我不能把它放在 ob 中。我需要得到它的第 0 个索引。

同样，当我们检查 done 的条件时，如果我们有一个列表，我们需要检查一个索引是假还是真，看看它是否结束。这是因为，在 python 或者大多数语言中，

```
bool([False, False])
```

返回 True。

在我的例子中，我创造了一个环境，如果一面完成了，所有的面都完成了。所以，总的来说，我做到了

```
def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if type(done) == list and done[0]:#For custom environments
            ob = env.reset()
            ob = ob[0]
        if type(done) != list and done:#For non-custom environments
            ob = env.reset()
        return ob, reward, done, info
```

就是这样！我基本上分成了两种情况，一种是我的游戏，一种是完成后返回的列表，另一种是通常的健身房环境。

我很确定我在这里可能有过于复杂的东西，所以如果你认为有些地方实现得不好，请告诉我，因为我喜欢学习。

3.渲染游戏

在这里，我只是使用了 render 函数，对环境做了与 reset 相同的事情(将渲染的图像放在一个列表中),并将 get_image 函数改为

```
def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        if self.sides > 1:
            imgs += [[None] for _ in range(len(self.remotes)*(self.sides-1))]
            imgs = self.tactic_game_fix_results(imgs)
            imgs = zip(*imgs)
            imgs = imgs.__next__()
        return imgs
```

4.这是稍微无关的，但由于**kwargs 用于初始化游戏，我对我的环境做了修改

环境是通过中的代码获得的

```
self.entry_point(**_kwargs)
```

其中 entrypoint 是 registeration.py 中的模块！基本上，这是用除了 env_id 参数之外的所有参数来初始化函数。

此外，在我的类的 __init__ 函数中，我这样做是为了使 kwargs 成为我的属性，因为在我的环境中，我已经用(argparse)做了类似的事情！你做这件事的方式，是我从[到这里](https://stackoverflow.com/questions/5624912/kwargs-parsing-best-practice)(都是迈克·刘易斯的功劳！)是

```
for k,v in kwarg.iteritems():
   setattr(self, k, v)
```

老实说，这很聪明！

但是，自从健身房。Envs 函数有一个重要的方法叫做 seeds，它碰巧和我在 build_env 函数中使用的一个参数同名

```
args_dict = vars(args)
del args_dict["seed"]
env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations, env_kwargs=args_dict)
```

在我运行 make_vec_env 之前

我能够成功地使用这个方法，因为之前种子参数被保存到种子变量中，该变量作为

```
seed = args.seed
```

# 一些其他的小细节

1.  这可能有点小，但我认为为了全面，我应该包括它。还记得我们包装环境和奖励等级的监视器包装吗？碰巧的是，每次我们调用 step in the runner.py 时，这些函数都会将奖励相加或乘以奖励。所以，因为我们没有返回任何奖励，这有点复杂，因为它会产生错误。因此，我需要检查它是否没有，但总的来说，我能够修复它，所以它很好！
2.  我注意到 gym 不知道我的 id，所以我需要在 run.py 中手动导入我的自定义环境来注册它。有点意思！也许有更好的方法，但现在，我会用这个！

# 最终结果

最后，我能够体面地开始训练了。我还没有运行一个全面的测试，所以仍然可能有错误，但目前为止看起来不错！

![](img/5a5b631a9a0cef9fab341406d52b1a64.png)

大约 10 分钟的训练后

以上是渲染的结果！要查看代码，请点击查看[！](https://github.com/isamu-isozaki/baseline-selfplay)