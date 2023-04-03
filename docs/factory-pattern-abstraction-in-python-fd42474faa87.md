# Python 中的工厂模式和抽象

> 原文：<https://medium.com/analytics-vidhya/factory-pattern-abstraction-in-python-fd42474faa87?source=collection_archive---------15----------------------->

Python 似乎是一种非常有趣的语言，一切尽在掌握。你可以写一些有用的代码，或者用流行的、受人喜爱的概念，比如可靠的、干净的代码和设计模式，写一些漂亮的代码。我不会把这篇文章写得太长，从现在开始，我会试着写一些关于 Python 的简单概念。在这篇文章中，我将谈论[工厂模式](https://en.wikipedia.org/wiki/Factory_method_pattern)，我们如何在 Python 中实现它，以及如何创建[抽象](https://en.wikipedia.org/wiki/Abstraction_(computer_science))来使事情变得更简单。

假设我们有一个音频播放器，我们可以播放 wav 和 mp3 格式。因此，基于参数`wav`或`mp3`，我们加载文件并播放它们。先做个界面吧。

```
from abc import ABC, abstractmethodclass AudioPlayer(ABC): @abstractmethod
    def load(self, file: str) -> (str):
        pass @abstractmethod
    def play(self) -> (str):
        pass
```

我已经使用了`abc`包来实现正式接口的概念。`@abstractmethod` decorator 暗示这些方法应该被具体的类覆盖。所以现在让我们来制造玩家。

```
class Mp3Player(AudioPlayer): def __init__(self):
        self.format = "mp3"
        self.file = None def load(self, file: str) -> (str):
        self.file = file
        return f"Loading {self.format} file named {file}" def play(self) -> (str):
        return f"playing {self.file}"class WavPlayer(AudioPlayer): def __init__(self):
        self.format = "wav"
        self.file = None def load(self, file: str) -> (str):
        self.file = file
        return f"Loading {self.format} file named {file}" def play(self) -> (str):
        return f"playing {self.file}"
```

所以我们有了`Mp3Player`和`Wavplayer`。他们实现了方法`load`和`play`。这两个类在这里是相同的，但是在实际的实现中，负载应该是不同的，可能游戏也是如此。现在是创建工厂的时候了。下面是 Python 的神奇之处！

```
player_factory = {
    'mp3': Mp3Player,
    'wav': WavPlayer
}
```

这太神奇了！你可以在字典中映射类，如此简单！在其他语言中，您可能必须编写几个 switch cases 或 if-else。现在可以直接用这个工厂调用我们的 load 来玩了。这在 Python 中被称为调度器。

```
mp3_player = player_factory['mp3']()
print(mp3_player.load("creep.mp3"))
print(mp3_player.play())wav_player = player_factory['wav']()
print(wav_player.load("that's_life.wav"))
print(wav_player.play())
```

看看我们如何基于参数初始化一个类！`mp3_player = player_factory[‘mp3’]()` —这真的很酷。所以整个代码看起来像这样—

```
from abc import ABC, abstractmethodclass AudioPlayer(ABC): @abstractmethod
    def load(self, file: str) -> (str):
        raise NotImplementedError @abstractmethod
    def play(self) -> (str):
        raise NotImplementedErrorclass Mp3Player(AudioPlayer): def __init__(self):
        self.format = "mp3"
        self.file = None def load(self, file: str) -> (str):
        self.file = file
        return f"Loading {self.format} file named {file}" def play(self) -> (str):
        return f"playing {self.file}"class WavPlayer(AudioPlayer): def __init__(self):
        self.format = "wav"
        self.file = None def load(self, file: str) -> (str):
        self.file = file
        return f"Loading {self.format} file named {file}" def play(self) -> (str):
        return f"playing {self.file}"player_factory = {
    'mp3': Mp3Player,
    'wav': WavPlayer
}mp3_player = player_factory['mp3']()
print(mp3_player.load("creep.mp3"))
print(mp3_player.play())wav_player = player_factory['wav']()
print(wav_player.load("that's_life.wav"))
print(wav_player.play())
```

现在你可以问如果用户在`player_factory`初始化中给`mp4`，会发生什么。好的，代码会崩溃。在这里，我们可以做一个抽象，隐藏所有创建类和验证参数的复杂性。

```
class AudioPlayerFactory: player_factory = {
        'mp3': Mp3Player,
        'wav': WavPlayer
    } @staticmethod
    def make_player(format: str):
        if format not in AudioPlayerFactory.player_factory:
            raise Exception(f"{format} is not supported")
        return AudioPlayerFactory.player_factory[format]()
```

现在我们可以使用`AudioPlayerFactory`来加载和播放。

```
mp3_player = AudioPlayerFactory.make_player('mp3')
print(mp3_player.load("creep.mp3"))
print(mp3_player.play())wav_player = AudioPlayerFactory.make_player('wav')
print(wav_player.load("that's_life.wav"))
print(wav_player.play())mp4_player = AudioPlayerFactory.make_player('mp4')
print(mp4_player.load("what_a_wonderful_life.mp4"))
print(mp4_player.play())
```

您将看到 mp4 文件的异常。你可以用自己的方式处理。所以新的代码是—

```
from abc import ABC, abstractmethod class AudioPlayer(ABC): @abstractmethod
    def load(self, file: str) -> (str):
        raise NotImplementedError @abstractmethod
    def play(self) -> (str):
        raise NotImplementedError class Mp3Player(AudioPlayer): def __init__(self):
        self.format = "mp3"
        self.file = None def load(self, file: str) -> (str):
        self.file = file
        return f"Loading {self.format} file named {file}" def play(self) -> (str):
        return f"playing {self.file}" class WavPlayer(AudioPlayer): def __init__(self):
        self.format = "wav"
        self.file = None def load(self, file: str) -> (str):
        self.file = file
        return f"Loading {self.format} file named {file}" def play(self) -> (str):
        return f"playing {self.file}"class AudioPlayerFactory: player_factory = {
        'mp3': Mp3Player,
        'wav': WavPlayer
    } @staticmethod
    def make_player(format: str):
        if format not in AudioPlayerFactory.player_factory:
            raise Exception(f"{format} is not supported")
        return AudioPlayerFactory.player_factory[format]()mp3_player = AudioPlayerFactory.make_player('mp3')
print(mp3_player.load("creep.mp3"))
print(mp3_player.play())wav_player = AudioPlayerFactory.make_player('wav')
print(wav_player.load("that's_life.wav"))
print(wav_player.play())mp4_player = AudioPlayerFactory.make_player('mp4')
print(mp4_player.load("what_a_wonderful_life.mp4"))
print(mp4_player.play())
```

希望这对你设计工厂有所帮助。还有另一种方法来隐藏工厂包的复杂性。我将很快讨论这个问题。掌声将不胜感激。