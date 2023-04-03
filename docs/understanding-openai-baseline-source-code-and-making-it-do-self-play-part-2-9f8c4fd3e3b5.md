# 了解 OpenAI 基线源代码，让它做自玩！第二部分

> 原文：<https://medium.com/analytics-vidhya/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-2-9f8c4fd3e3b5?source=collection_archive---------9----------------------->

![](img/0c4e24ae4dd215d07c5cf52498b2ba0f.png)

在上一节中，我们一直到 train 函数，并在实际构建环境之前停下来。要查看，请点击[此处](/analytics-vidhya/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-1-9f30085a8c16)！

# 构建 _ 环境

在下一行，我们看到

```
env = build_env(args)
```

那么，让我们来看看 build_env 函数！

```
def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed env_type, env_id = get_env_type(args) if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed,             wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size) else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config) flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations) if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True) return env
```

首先，

```
ncpu = multiprocessing.cpu_count()
if sys.platform == 'darwin': ncpu //= 2
nenv = args.num_env or ncpu
alg = args.alg
seed = args.seed
```

似乎在初始化参数。这里，一个非常常见但有趣的技巧是使用关键字 or 来利用 nenv 变量。一旦你考虑到“或”是做什么的，这是非常直观的。首先，要注意的一个基本事实是，如果其中一个参数为真，or 语句将返回 true。因此，如果 args.num_env 不是 None，它将被设置为 nenvs，否则它将返回 ncpu(根据[这里的](https://stackoverflow.com/questions/31344582/python-multiprocessing-cpu-count-returns-1-on-4-core-nvidia-jetson-tk1)，它是在线 cpu 的数量)！这是一个让你的代码更短的巧妙方法！

接下来，我们看到

```
env_type, env_id = get_env_type(args)if env_type in {'atari', 'retro'}:
```

因为我们可以从参数中指定 env_type，并且因为我们的环境是自定义的，所以我们可以跳过这个 if 语句的内容，直接进入 else 部分。

```
else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config) flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1,        seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations) if env_type == 'mujoco':
            env = VecNormalize(env, use_tf=True)
```

首先，什么是允许软放置？根据[这里的](https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac)，它允许你可以在特定的 GPU 上运行操作。例如，你可以做

```
with tf.device('/gpu:0'):
```

具体分配到第 0 个 gpu 哪个有意思。

突出的下一行是

```
env = make_vec_env(env_id, env_type, args.num_env or 1,        seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations
```

现在，让我们看看 make_vec_env 是做什么的！

# 制造 _ 车辆 _ 环境

make_vec_env 驻留在 cmd_utils.py 中

```
def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer
        ) set_global_seeds(seed)
     if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
     else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])
```

如果我们在运行时想要多个环境，我就是这种情况！因此，我们将只查看 SubprocVecEnv。但我认为公平地说，在对环境的期望方面，它将与 DummyVecEnv 非常相似。

在检查它的作用之前，让我们先检查 make_thunk。这是一个有趣的函数，因为它返回一个带有传入参数的函数。它的方式是下面的方法

```
def a(args):
    return lambda : b(args)
```

我觉得这很酷！无论如何，我们不要分心，看看返回的函数:make_env。

# 制作 _ 环境

```
def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank) wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    if env_type == 'atari':
        env = make_atari(env_id)
    elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id,   max_episode_steps=10000,   use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    else:
        env = gym.make(env_id, **env_kwargs) if flatten_dict_observations and   isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env,  dict_keys=list(keys)) env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir,       str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True) if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    elif env_type == 'retro':
        if 'frame_stack' not in wrapper_kwargs:
            wrapper_kwargs['frame_stack'] = 1
        env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs) if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env) if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale) return env
```

对于前两行，

```
 if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)
```

我不是特别确定什么是初始化式，但是对于我们之前看过的所有函数，初始化式都被设置为 None，所以我们现在跳过它！

下一段有趣的代码是

```
if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
```

基本上，这里发生的事情是，我的自定义环境将被加载到这里。这是通过删除 env_id 中的所有 past:并返回 before 部分来实现的！另一方面，env_id 被设置为“:”后面的部分。这是通过添加一个句点来实现的，该句点基本上匹配除换行符之外的所有字符，而*表示它匹配 0 个或更多前面的标记(。).在这里执行 re.sub 会将这些模式突出显示的部分替换为第二个参数''，nothing。我发现查看 regex 很有趣，因为我对它不是特别熟悉，试图找到它做什么就像一个小难题。够了！让我们继续

```
if env_type == 'atari':
        env = make_atari(env_id)
elif env_type == 'retro':
        import retro
        gamestate = gamestate or retro.State.DEFAULT
        env = retro_wrappers.make_retro(game=env_id,    max_episode_steps=10000,     use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
else:
        env = gym.make(env_id, **env_kwargs)
```

因为我们的环境既不是 atari 也不是 retro，所以运行 gym.make 功能。这个函数是什么？

# 制造

我花了一段时间才找到它，但在 gym . make . registration 我找到了它！

```
registry = EnvRegistry()def register(id, **kwargs):
    return registry.register(id, **kwargs)def make(id, **kwargs):
    return registry.make(id, **kwargs)
```

这个 make 函数运行函数

```
def make(self, path, **kwargs):
        if len(kwargs) > 0:
            logger.info('Making new env: %s (%s)', path, kwargs)
        else:
            logger.info('Making new env: %s', path)
        spec = self.spec(path)
        env = spec.make(**kwargs)
        # We used to have people override _reset/_step rather than
        # reset/step. Set _gym_disable_underscore_compat = True on
        # your environment if you use these methods and don't want
        # compatibility code to be invoked.
        if hasattr(env, "_reset") and hasattr(env, "_step") and not       getattr(env, "_gym_disable_underscore_compat", False):
            patch_deprecated_methods(env)
        if (env.spec.max_episode_steps is not None) and not spec.tags.get('vnc'):
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)
        return env
```

第一，什么是 kwargs？这是 python 中比较好的特性之一，如果你在调用一个函数时有一个变量名称=变量形式的参数列表，比如(向极客的极客们致敬)

```
**def** myFun(arg1, ******kwargs):
   **for** key, value **in** kwargs.items():
        print ("%s == %s" **%**(key, value))# Driver codemyFun("Hi", first **=**'Geeks', mid **=**'for', last**=**'Geeks')
```

它会打印出来

```
last == Geeks
mid == for
first == Geeks
```

它的名字就是这么来的！关键字参数，kwargs。您可以使用*args 来获得通常的参数，但这应该在 kwargs 之前！

然后，这个函数在 env 文件的路径上运行 spec 函数。

# 投机

```
def spec(self, path):
        if ':' in path:
            mod_name, _sep, id = path.partition(':')
            try:
                importlib.import_module(mod_name)
            # catch ImportError for python2.7 compatibility
            except ImportError:
                raise error.Error('A module ({}) was specified for the environment but was not found, make sure the package is installed with `pip install` before calling `gym.make()`'.format(mod_name))
        else:
            id = pathmatch = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to look up malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id.encode('utf-8'), env_id_re.pattern))try:
            return self.env_specs[id]
        except KeyError:
            # Parse the env name and check to see if it matches the non-version
            # part of a valid env (could also check the exact number here)
            env_name = match.group(1)
            matching_envs = [valid_env_name for valid_env_name, valid_env_spec in self.env_specs.items()
                             if env_name == valid_env_spec._env_name]
            if matching_envs:
                raise error.DeprecatedEnv('Env {} not found (valid versions include {})'.format(id, matching_envs))
            else:
                raise error.UnregisteredEnv('No registered env with id: {}'.format(id))
```

规范首先看路径参数。如果它包含一个模块(includes:)，它将尝试导入它并将 id 部分放入 id 变量中。否则，它只是将 id 变量设置为 path。

然后，

```
match = env_id_re.search(id)
```

在哪里

```
env_id_re = re.compile(r'^(?:[\w:-]+\/)?([\w:.-]+)-v(\d+)$')
```

现在，是时候来点正则表达式了！阅读[此处](https://docs.python.org/3/library/re.html)跟随前进！^运算符匹配字符串的开头，然后是(？:…)匹配括号内的所有内容。这将匹配多个(+)单词(\w)、冒号和减号。接下来是/。？这表示前面模式的 0 或 1 次重复。最后，它将再次匹配单词、冒号和减号，后面跟一个 v 和最后以$结尾的小数(\d+)。$表示这是该行的结尾..例如，如果 id 有一个模式

```
env_name-v0
```

这将是一场比赛！因此，它必须是环境名和版本的形式

因此我们知道 id 必须是什么形式。代码中的下一行试图在 env_specs 变量中搜索 id。这应该添加在 make 函数之前调用的函数中，register 函数将通过

```
def register(id, **kwargs):
    return registry.register(id, **kwargs)
```

和

```
def register(self, id, **kwargs):
        if id in self.env_specs:
            raise error.Error('Cannot re-register id: {}'.format(id))
        self.env_specs[id] = EnvSpec(id, **kwargs)
```

在 EnvRegistry 类中。因此，返回 id 的 EnvSpec。

根据注释，EnvSpec 类存储“特定环境实例的规范”。使用
登记官方评估的参数在 init 函数中，我们在 EnvSpec 的 __init__ 函数中看到了这一点

```
def __init__(self, id, entry_point=None, reward_threshold=None,    kwargs=None, nondeterministic=False, tags=None, max_episode_steps=None):
        self.id = id
        # Evaluation parameters
        self.reward_threshold = reward_threshold
        # Environment properties
        self.nondeterministic = nondeterministic
        self.entry_point = entry_pointif tags is None:
            tags = {}
        self.tags = tagstags['wrapper_config.TimeLimit.max_episode_steps'] = max_episode_steps

        self.max_episode_steps = max_episode_steps# We may make some of these other parameters public if they're
        # useful.
        match = env_id_re.search(id)
        if not match:
            raise error.Error('Attempted to register malformed environment ID: {}. (Currently all IDs must be of the form {}.)'.format(id, env_id_re.pattern))
        self._env_name = match.group(1)
        self._kwargs = {} if kwargs is None else kwargs
```

# 回到注册表 make 函数

下一行执行

```
env = spec.make(**kwargs)
```

所以我们来看看 Spec 的 make 函数。

# 规格制造功能

```
def make(self, **kwargs):
        """Instantiates an instance of the environment with appropriate kwargs"""
        if self.entry_point is None:
            raise error.Error('Attempting to make deprecated env {}. (HINT: is there a newer registered version of this env?)'.format(self.id))
        _kwargs = self._kwargs.copy()
        _kwargs.update(kwargs)
        if callable(self.entry_point):
            env = self.entry_point(**_kwargs)
        else:
            cls = load(self.entry_point)
            env = cls(**_kwargs) # Make the enviroment aware of which spec it came from.
        env.unwrapped.spec = selfreturn env
```

首先，获取入口点并检查它是否是有效的入口点。根据评论，它应该是这样的形式

```
The Python entrypoint of the environment class (e.g. module.name:Class)
```

或者它可以只是模块名

然后，它通过调用 self.entry_point 或者通过调用

```
def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn
```

然后调用那个函数！

然后用关键字参数运行它。然后它设置 self.unwrapped_spec，但我不完全确定这是做什么的。

# 注册表生成功能

完成后

```
if hasattr(env, "_reset") and hasattr(env, "_step") and not getattr(env, "_gym_disable_underscore_compat", False):
            patch_deprecated_methods(env)
```

首先，它检查 env 是否有方法 _reset 和 _step，我认为这是内置在 gym.Env 中的。因此，我怀疑 getattr(env，" _ gym _ disable _ 下划线 _compat "，False)的默认输出是 False。我们可以通过简单地运行

```
import gym
getattr(gym.Env, "_gym_disable_underscore_compat", False)
```

它确实返回了 False！我去健身房了。因为这是我们定制环境的基础。就像我从体育馆继承的一样。通过做来包围

```
class Game_Env(gym.Env):
    def __init__
...
```

因此，下一行

```
patch_deprecated_methods(env)
```

会被挤兑。这基本上将 env.reset 设置为 env。_reset 和 env.step to env。_step 等等。但是等等，那么我们定义的阶跃函数会发生什么呢？为了检查这个函数是否实际执行，我运行了

```
hasattr(gym.Env, "_reset")
```

返回 False。因此，该语句不会针对我的环境运行！

最后，添加如下时间限制

```
if (env.spec.max_episode_steps is not None) and not spec.tags.get('vnc'):
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env, max_episode_steps=env.spec.max_episode_steps)
```

我发现他们这样做的方式非常有趣，因为它有助于抽象环境。当您查看 gym.wrappers.time_limit 时，TimeLimit classes only 函数似乎正在跟踪它处于哪个步骤，同时还执行环境的步骤，如下所示！

```
class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
```

它特别好，因为每个包装器都增强了步进和重置功能！我想从现在开始我会做这样的事情，因为它看起来很酷。目前，我的环境的代码在 1000 行的情况下不是特别好，所以我计划使用这种技术使我的代码更整洁。

反正过了这一关，环境归环境！

# 返回 make_env

下一行代码是

```
if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
```

但是既然我的观察空间都是空间。框中，我可以跳过这一行，但由于我有点好奇字典是如何“展平”的，让我们来检查一下！if 语句的内容是

```
env = FlattenObservation(env)
```

我在体育馆的包装纸里找到的，没错

```
import numpy as np
import gym.spaces as spaces
from gym import ObservationWrapperclass FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env) flatdim = spaces.flatdim(env.observation_space)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(flatdim,), dtype=np.float32) def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)
```

所以，基本上，它使用空间中定义的函数将观察空间改变为一个盒子。这个模块导入了 utils，我在里面找到了 flatten 和 flatdim

对于所有可能的观察空间，flatdim 返回单个整数形式的整数，如下所示

```
def flatdim(space):
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return int(space.n)
    elif isinstance(space, Tuple):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError
```

所以，在应用这个简洁的包装器后，self.observation_space 将变成一维空间！我不确定这对卷积神经网络来说是不是一件好事，事实上这是一个图像问题，但无论如何，我觉得这很有趣。

同样，flatten 函数也是这样做的，它将观察空间展平为一维数组。

然后，我们去下一行！

```
env.seed(seed + subrank if seed is not None else None)
env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)
```

就是跑。我怀疑 env.seed 只是为概率分布等设置了种子。然而，Monitor 是做什么的呢？在 baselines/bench 中查看 monitor.py 之后，似乎这个包装器所做的就是在每一步，将奖励写入一个 csv 和 json 文件，以“monitor.json”和“monitor.csv”结尾！

接下来，

```
if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)
```

就是跑。这个 ClipActionsWrapper 的动作函数被定义为

```
np.clip(action, self.action_space.low, self.action_space.high)
```

因此，它将动作限制在设定的最大值和最小值。我喜欢他们用包装器来包装像这样的细节。它增强了可理解性，同时使代码更容易调试。

最后，

```
if reward_scale != 1:
        env = retro_wrappers.RewardScaler(env, reward_scale)return env
```

这个包装器所做的只是将奖励乘以 reward_scale

```
def reward(self, reward):
        return reward * self.scale
```

现在，最终，env 被返回。

在我们讨论完 make_env 之后，让我们休息一下，在下一篇文章中讨论 SubprocVecEnv！

# 然后

在下一篇文章中，我们将了解 SubprocVecEnv 并接触多处理！