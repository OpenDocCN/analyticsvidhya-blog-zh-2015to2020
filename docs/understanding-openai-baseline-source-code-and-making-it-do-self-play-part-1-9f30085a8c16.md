# 了解 OpenAI 基线源代码，让它做自玩！第一部分

> 原文：<https://medium.com/analytics-vidhya/understanding-openai-baseline-source-code-and-making-it-do-self-play-part-1-9f30085a8c16?source=collection_archive---------8----------------------->

![](img/0c4e24ae4dd215d07c5cf52498b2ba0f.png)

# 最终结果

要查看最终结果，请查看这篇[文章！](/analytics-vidhya/making-a-self-play-environment-for-openai-gym-23486bc44d6f)

# 通常的程序

当我们想要将一个环境应用于这些基线算法时，通常的程序是首先创建一个环境，然后使它成为一个 OpenAI 健身房！正如[这篇好文章](/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)中所写的，这是通过构建一个文件结构来实现的，比如(摘自文章。全部归功于 Ashish)

```
gym-env/
  README.md
  setup.py
  gym_env/
    __init__.py
    envs/
      __init__.py
      env.py
```

其中 env.py 包含您的环境！这个环境的主要规范是它需要在表单中(再次感谢 Ashish！)

```
class Env(gym.Env):
  metadata = {'render.modes': ['human']}def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human', close=False):
```

而且需要一个观察空间和行动空间！这些变量应该定义为 self.action_space 和 self.observation_space。更多详情请阅读[的另一篇好文章](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)！但基本上，它们必须是 gym.spaces 中定义的类型之一。可以在这里找到列表(每个 python 文件对应一种类型的空间)。比如在[亚当的文章](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)中，他把自己的观察 _ 空间和行动 _ 空间定义为

```
*# Example when using discrete actions:*
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS) *# Example for using image as input:*
    self.observation_space = spaces.Box(low=0, high=255, shape=
                    (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
```

# 问题是

然而，我对这种方法感到的一个问题或困惑是，我不知道如何扩展这种方法来让一台机器与自己比赛。

在这一点上，我想我理解你如何处理传统的强化问题，例如保持一根杆子稳定或买卖股票，但我不知道如何处理有两个或更多方面的环境，其中每一方面都有自己的观察结果，对同一行动有不同的结果。我想要的是一个让我的电脑自我对抗的方法！为此，我选择查看源代码！

如果你想按照说明去做，就去做

```
git clone [https://github.com/openai/baselines.git](https://github.com/openai/baselines.git)
```

获取基线的代码！

# 哪个文件正在运行？

当您查看 [openAI 基线](https://github.com/openai/baselines)的自述文件时，您会发现为了训练一个环境，命令

```
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```

已使用。python 命令的-m 部分基本上是-module 的缩写。所以，

```
python -m baselines.run
```

相当于

```
cd baselines
python run.py
```

因此，让我们看看 run.py 的内部，看看它是如何运行的！

# run.py

如果我们进入 baselines 文件夹并进入 run.py，我们会看到运行的第一个函数是

```
if __name__ == '__main__':
    main(sys.argv)
```

还是去主函数吧！

# 主要的

```
def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0) arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    #import_module(args.custom_env_module)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[]) model, env = train(args, extra_args) if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path) if args.play:
        logger.log("Running trained model")
        obs = env.reset() state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,)) episode_rew = np.zeros(env.num_envs) if isinstance(env, VecEnv) else np.zeros(1)
        while True:
            if state is not None:
                actions, _, state, _ = model.step(obs,S=state, M=dones)
            else:
                actions, _, _, _ = model.step(obs) obs, rew, done, _ = env.step(actions)
            episode_rew += rew
            env.render()
            done_any = done.any() if isinstance(done, np.ndarray) else done
            if done_any:
                for i in np.nonzero(done)[0]:
                    print('episode_rew={}'.format(episode_rew[i]))
                    episode_rew[i] = 0 env.close()
    return model
```

现在，既然这是一个大函数，就让我们一步一步来！在第一行，我们看到

```
arg_parser = common_arg_parser()
```

# 公共参数解析器

这个 common_arg_parser 驻留在 cmd_util.py 中，该文件来自 baselines 文件夹中的 common 文件夹。就目录而言，该文件可以在 baselines/common/cmd_util.py 中找到

```
def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--alg', help='Algorithm', type=str, default='ppo2')
    parser.add_argument('--num_timesteps', type=float, default=1e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)
    parser.add_argument('--log_path', help='Directory to save learning curve data.', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    return parser
```

这个函数的作用是设置你可以调用的参数。游戏步骤，有趣的是，默认设置为 100 万。因此，我们可以假设一旦完成，环境就会重置！我发现的另一个有趣的功能，除了运行多种环境的能力，是你可以保存你的游戏到一个视频，这使得它更容易看到进展！

# 返回主页

下一行的主要功能是

```
args, unknown_args = arg_parser.parse_known_args(args)
extra_args = parse_cmdline_kwargs(unknown_args)
```

这是使用 arg_parser 的代码，arg _ parser 是一个库，它使您可以访问您在命令行中设置的参数

```
--arg arg_content
```

以便 args.arg 等于 arg_content

因此，args 变量包含所有的参数和它们的值，虽然在[文档](https://docs.python.org/3.4/library/argparse.html)中并不完全清楚，但是从这个变量的命名方式来看，我们可以假设 unknown_args 包含 common_arg_parser 中的参数。对于下一行，变量 extra_args 由函数 parse_cmdline_kwargs 初始化

# parse_cmdline_kwargs

该函数的代码是

```
def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v): assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v return {k: parse(v) for k,v in parse_unknown_args(args).items()}
```

这个函数的作用是试图改变参数(如果可以的话，将参数的内容改为实际的可执行关键字！然后把它放在字典里，关键是你的论点。这是

```
--args
```

段。有趣的是在 python 中你可以

```
eval(string)
```

将字符串转换为表达式。挺有意思的。

这个解析 unknown_args 使它成为一个字典。

# 下一行

```
if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
    rank = 0
    configure_logger(args.log_path)
else:
    rank = MPI.COMM_WORLD.Get_rank()
    configure_logger(args.log_path, format_strs=[])
```

现在，什么是 MPI？上面，我们可以找到定义为

```
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
```

在快速的 google 搜索之后，很明显这个模块是用于并行处理的！MPI。COMM_WORLDGet_rank()将等级设置为它。然而，由于 rank 关键字除非为 0，否则不会做任何事情，并且它似乎不会以显著的方式影响代码，所以我们将直接使用 configure_logger！如果读者愿意，可以进一步研究这个函数，但是总的来说，顾名思义，它的作用是设置文件登录的位置！

# 火车

下一行最后是

```
model, env = train(args, extra_args)
```

现在，火车内部的功能是什么？

```
def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type)) total_timesteps = int(args.num_timesteps)
    seed = args.seed learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args) env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x %     args.save_video_interval == 0, video_length=args.save_video_length) if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type) print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs)) model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    ) return model, env
```

哇！这本身就是一个相当长的函数。在前四行中，

```
env_type, env_id = get_env_type(args)
print('env_type: {}'.format(env_type))total_timesteps = int(args.num_timesteps)
seed = args.seed
```

我们看到环境的 id 和类型已经设置好了。我不太清楚这些变量是什么。因此，我们将看到我们将继续前进！时间步长和种子由自变量初始化。

```
learn = get_learn_function(args.alg)
alg_kwargs = get_learn_function_defaults(args.alg, env_type)
alg_kwargs.update(extra_args)env = build_env(args)
```

现在，从 get_learn_function 获得一个函数 learn。这个的代码是

```
def get_learn_function(alg):
    return get_alg_module(alg).learn
```

什么是 get_alg_module？

```
def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))return alg_module
```

它基本上是通过调用 import_module 函数首先导入，或者至少尝试导入基线中的算法。如果我对模块的理解正确的话，这将返回一个对象，其中上述 python 文件中的所有全局变量都是它的属性。例如，由于在所有算法模块的全局空间中有一个名为 learn 的函数(如基线中的 PPO 2 . py ), alg _ module . learn 应该返回该函数。

这样，get_learn_function 就得到算法的学习函数。

然后，调用 get_learn_function_defaults。该功能定义如下

```
def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs
```

这选择了网络应该具有的参数。例如，在 ppo2 的默认模块中，我们发现

```
def mujoco():
    return dict(
        nsteps=2048,
        nminibatches=32,
        lam=0.95,
        gamma=0.99,
        noptepochs=10,
        log_interval=1,
        ent_coef=0.0,
        lr=lambda f: 3e-4 * f,
        cliprange=0.2,
        value_network='copy'
    )def atari():
    return dict(
        nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=0.1,
    )def retro():
    return atari()
```

getattr 在这里是一个很好的函数，因为即使 env_type 是一个字符串，它仍然可以执行以返回该函数的 dict！

这里需要注意的一点是，我可能会为做 AI 添加自己的字典返回函数。或者，这就是为什么有额外的参数，因为在下一行中，我们将字典更新为

```
alg_kwargs.update(extra_args)#adds arguments
```

这是非常用户友好的。

# 然后

在下一篇文章中，我将讨论环境是如何形成的，并对 OpenAI 代码的某些部分(特别是包装器)感到惊讶。点击这里查看！