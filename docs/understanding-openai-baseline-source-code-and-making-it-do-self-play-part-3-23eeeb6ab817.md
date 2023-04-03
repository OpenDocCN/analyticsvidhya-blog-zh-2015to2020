# 了解 OpenAI 基线源代码，让它做自玩！第三部分

> 原文：<https://medium.com/analytics-vidhya/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-3-23eeeb6ab817?source=collection_archive---------14----------------------->

![](img/0c4e24ae4dd215d07c5cf52498b2ba0f.png)

在之前的文章中，我们介绍了 OpenAI 是如何构建的，以及我觉得有趣的一些部分。你可以查看第一部分和第二部分[这里](/analytics-vidhya/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-1-9f30085a8c16)和[这里](/@isamu.website/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-2-9f8c4fd3e3b5)！

# SubprocVecEnv _ _ init _ _:深入了解多处理

__init__ 函数如下所示

```
class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """
    def __init__(self, env_fns, spaces=None, context='spawn', in_series=1):
        """
        Arguments: env_fns: iterable of callables -  functions that create   environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a  single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series" self.nremotes = nenvs// in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close() self.remotes[0].send(('get_spaces_spec', None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x

        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)
```

第一行用于初始化

```
self.waiting = Falseself.closed = Falseself.in_series = in_seriesnenvs = len(env_fns)
```

然后，

```
assert nenvs % in_series == 0, "Number of envs must be divisible by number of envs to run in series"self.nremotes = nenvs// in_series
```

这里，in_series 变量开始发挥作用。nenvs 变量是环境的数量，但是 in_series 变量，正如文本所说，似乎将这些环境划分为 in_series 多个进程，并为每个进程分配 nenvs // in_series 多个环境，反之亦然！

```
env_fns = np.array_split(env_fns, self.nremotes)
```

在这里，我们看到它是反过来的。像 np.array_split 函数一样，有 nenv//in_series 多个进程，每个进程在 in _ series 多个环境中运行。这个函数只是将它们分开，并让基本索引选择要查看的部分。举个例子，

```
np.array_split(np.arange(6), 2)
```

will 返回[array([0，1，2])，array([3，4，5])]。有趣的是，外部是一个列表，而不是一个 np 数组。然后，

```
ctx = mp.get_context(context)
```

就是跑。这位议员来自

```
import multiprocessing as mp
```

根据[文档](https://docs.python.org/3/library/multiprocessing.html)，很明显，当你使用一个上下文时，你可以把它们当作一个单独的对象。我不太清楚这是什么意思，但我们继续吧！

接下来，

```
self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.nremotes)])
```

就是跑。一、什么是 ctx。管道()？让我们检查一下！根据相同的[文档](https://docs.python.org/3/library/multiprocessing.html)，管道构成如下两个连接的对象

```
parent_conn, child_conn = Pipe()
```

从这里你可以做的是，他们可以进一步互相交谈！你这样做的方法是

parent_conn.send 向 child_conn 发送消息，反之亦然！

另一方面，parent_conn.recv()从子节点获取消息，反之亦然。

因此，在这里，self.remotes 对应于父节点的列表，而 self.work_remotes 对应于相应子节点的列表。

然后，

```
self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
```

就是跑。

将流程运行时运行的函数作为目标。这可以通过调用。运行()或 by。start()调用在一个单独的进程中运行！

参数只是目标函数的参数。在这种情况下，这就是工人。

CloudpickleWrapper 定义如下！

```
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """def __init__(self, x):
        self.x = xdef __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
```

它主要做的是将 env_fn 函数保存在 x 中，然后，__getstate__ function 返回 self.x 的序列化表示，这基本上意味着，如果将从 cloudpickle.dumps(self.x)返回的东西保存到一个文件中，以后可以通过调用 def __setstate__ 来加载它，以检索所有的属性等等！根据[这里的](https://pypi.org/project/cloudpickle/)，cloudpickle 被用于 __getstate__ 的原因是因为它支持更多的东西。

然后，

```
for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            with clear_mpi_env_vars():
                p.start()
```

就是跑。从计算机科学的角度来看，守护进程意味着后台进程。所以，即使它崩溃了，也不会有什么不好的事情发生，这是有道理的！

对于 clear_mpi_env_vars，根据注释，

```
"""from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessingProcesses."""
```

这很有趣。然后，最后，

```
p.start()
```

环境开始了！

```
for remote in self.work_remotes:
            remote.close()self.remotes[0].send(('get_spaces_spec', None))
observation_space, action_space, self.spec = self.remotes[0].recv().x
self.viewer = None
VecEnv.__init__(self, len(env_fns), observation_space, action_space)
```

我不明白为什么 close 方法在这里，如果有人知道，请告诉我！我认为这与初始化有关，但我不完全确定。

然后，从环境中检索规范，并调用 VecEnv 的 init 函数。

但这是一种新的叫法！我习惯于看到

```
class a:
    def __init___(self):
        self.a = "a"
    def hello(self):
        print(self.a)
class b(a):
    def __init__(self):
        super(b, self).__init__()
```

而不是函数中的直接一个. __init__。我不完全确定，但我认为这允许 a 的属性值被复制到 b，因为通常情况下，你不能这样做。

# VecEnv

```
class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = Nonemetadata = {
        'render.modes': ['human', 'rgb_array']
    }def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
```

既然是设置属性，我怀疑我是对的！现在，让我们来看看重置，步进和渲染功能！

```
def step(self, actions):
    """
    Step the environments synchronously.This is available for backwards compatibility.
    """
    self.step_async(actions)
    return self.step_wait()def render(self, mode='human'):
    imgs = self.get_images()
    bigimg = tile_images(imgs)
    if mode == 'human':
        self.get_viewer().imshow(bigimg)
        return self.get_viewer().isopen
    elif mode == 'rgb_array':
        return bigimg
    else:
        raise NotImplementedError
@abstractmethod
def reset(self):
    passraise NotImplementedError[@abstractmethod](http://twitter.com/abstractmethod)
def reset(self):
    pass
```

@abstractclass 是由 abc(抽象基类)模块提供的一个装饰器，它基本上是说这个方法需要被覆盖。

所以，基本上，对于每个步骤，运行 step_async 函数并返回 step_wait 函数。而这些函数都是在 SubprocVecEnv 中给出的！

# 步骤 _ 异步

```
def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
```

因此，远程将字符串“step”和动作发送给其子节点，并将变量 self.waiting 设置为 True。

# 步骤 _ 等待

```
def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos
```

我们看到遥控器从它孩子那里收到一条信息，这是环境的状态！我们看到它通过运行 _flatten_list 展平列表后，返回 obs，OBS，observations，rews，rewards，dones，是否结束，以及我推测是多余的 infos。

然后，经过一些处理，它返回它们！现在，让我们看看工人，实际上运行一切的功能(进程的目标)

但是，在继续之前，让我们看看 flatten_list！

# 扁平化 _ 列表

我必须说，当我第一次看到这个的时候，我很困惑。

```
def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])
    return [l__ for l_ in l for l__ in l_]
```

所以，我们先来看看 l_。l_ 是 l_ 的元素，因为它是为 l_ in l 而写的。因此，对于 l_ 的每个元素，都返回一个名为 l_ 的东西。这是 l_ 的元素，正如它在 l_ 中对 l_ 的描述。因此，我们可以看到，这是一个 2d 循环的简写，通过将所有行彼此相邻放置，将索引放入 1d 数组中。

# 工人

```
def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, infoparent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == 'reset':
                remote.send([env.reset() for env in envs])
            elif cmd == 'render':
                remote.send([env.render(mode='rgb_array') for env in envs])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(
CloudpickleWrapper((envs[0].observation_space, envs[0].action_space, envs[0].spec)))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        for env in envs:
            env.close()
```

首先是因为，

```
def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info
```

这似乎调用了我们环境的 step 函数，并返回观察结果、奖励、环境是否完成以及额外的信息并返回它们！

此外，在这里，如果环境完成，环境被重置，第一个观察结果被返回，这取代了很酷的观察结果！

然后，

```
parent_remote.close()
envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
```

我最初认为这会一直创造一个新的环境，但如果是这样的话，我认为任何形式的游戏都是不可能的。因此，我怀疑 __get_state__ 和 __set_state__ 方法在维护环境状态方面发挥了作用。但是我没有在任何地方看到它，所以我不能完全确定。

看起来像是工作程序运行 env 函数，当它完成时，它将数据发送到远程！这个函数中有很多关闭，这可能表明，它将暂时切断通信。例如，如果发送了两条消息，而孩子想一次读一条，我认为这类似于关闭通信，这样第二条消息还没有到达，读第一条消息然后等待下一条消息。但我不完全确定。

对我来说，另一件有趣的事情是，在 reset 函数中，观察值也需要被发送回来！所以我会这么做的！

现在，让我们回到 run.py 的 build_env！

# 构建 _ 环境

完成之后，build_env 只返回环境。那我们上去训练吧！

# 火车

build_env 完成后，

```
if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)
```

结束了。这个，只是用 gym.wrappers.monitoring 的 video_recorder 录视频。

然后，在网络设置如下之后，

```
if args.network:
        alg_kwargs['network'] = args.network
else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
```

型号设置为

```
model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )
```

这基本上是所有选择的算法的学习功能！

最后，返回模型和环境

```
return model, env
```

然后，最后，让我们进入学习功能，看看我们应该做什么！

# ppo2 学习功能

因为有很多算法，但我认为它们对环境的期望应该是相似的，所以我决定看看 ppo2 的学习功能！我选择 ppo2 是因为，嗯，我比其他人更了解它。

在 learn 函数中，我们看到的与环境交互的第一个实例是

```
obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
```

现在，什么是跑步者？它被定义为

```
runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
```

Runner 模块在 baselines/ppo2/runner.py 中定义

```
self.obs[:], rewards, self.dones, infos = self.env.step(actions)
```

在运行功能中。也

```
actions, values, self.states, neglogpacs = self.model.step(self.obs, S=self.states, M=self.dones)
```

在它之前。

总的来说，我们看到，虽然这是一次有趣的旅程，但我们已经回到了最本质的问题:我们应该做些什么来使这成为一个自我游戏的环境？

# 然后

在下一篇文章中，我将在这里讨论我是如何实现这个[的](/@isamu.website/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-4-a54e075386bf)！