# 简介配置

> 原文：<https://medium.com/analytics-vidhya/introducing-pyfiguration-4d306cbf6321?source=collection_archive---------31----------------------->

![](img/c8a696f685af7076668ba7840caca33b.png)

塞萨尔·卡利瓦里诺·阿拉贡在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

## Python 配置库，使配置接近您的代码。

有许多(很棒的)库可以用来配置 Python 脚本。你可以使用带有 [argparse](https://docs.python.org/3/library/argparse.html) 或 [Click](https://click.palletsprojects.com/en/7.x/) 的命令行选项，使用带有 [JSON 模式](https://json-schema.org/)的 JSON 文件，使用带有 [PyYAML](https://github.com/yaml/pyyaml) 的 YAML，或者使用老式的。带有 [configparser](https://docs.python.org/3/library/configparser.html) 的 INI 文件。我曾经在某个时间点使用过上述所有方法，但不知何故，我总是以不再有效或不再使用的配置条目结束，并且我一直打开代码来检查哪些选项是可用的。

配置与代码分离，这是一件好事，但是为了优化代码的使用，您应该能够快速显示哪些配置选项是可用的，允许的值是什么，以及它将如何影响您的代码。这就是我开始[py configuration](https://pyfiguration.readthedocs.io/en/latest/)的原因。

使用[p 配置](https://pyfiguration.readthedocs.io/en/latest/)您可以在代码中用装饰器定义配置选项。您将按预期使用这些配置，当您完成时,[p 配置](https://pyfiguration.readthedocs.io/en/latest/)命令行工具帮助记录所有配置选项。它甚至会检查配置文件是否有效，这对 CI/CD 管道非常有用！

[https://github.com/gijswobben/pyfiguration](https://github.com/gijswobben/pyfiguration)

# 说够了，秀点代码！

让我们从一个简单的例子开始，为一个 [Flask](https://flask.palletsprojects.com/en/1.1.x/) 服务器定义一些配置:

```
# script.pyfrom pyfiguration import conf
from flask import Flaskapp = Flask(__name__)@conf.add_int_field(
    "server.port",
    description="The port that the server will run on",
    default=5000,
    *minValue*=80,
    *maxValue*=65535
)
*def* startServer():
    port = conf["server"]["port"]
    app.run(*port*=port)if __name__ == "__main__":
    startServer()
```

在这个例子中，我们用 Flask 创建了一个简单的服务器，并用一个新的配置选项修饰了`startServer`方法。通过这种方式，我们保持配置接近代码。此外，我们还定义了一个默认值，因此当没有可用的配置文件时，我们将在端口 5000 上启动服务器。最后，我们希望确保服务器端口为 80 或更高，并且低于 65535(这是机器上最大的可用端口号)。

现在我们可以使用[py configuration](https://pyfiguration.readthedocs.io/en/latest/)命令行工具来检查这个脚本:

```
$ pyfiguration inspect script -s ./script.py
The following options can be used in a configuration file for the module './examples/simple/x.py':
server:
  port:
    allowedDataType: int
    default: 5000
    description: The port that the server will run on
    maxValue: 65535
    minValue: 80
    required: true
```

输出(YAML 格式)告诉我们在一个配置文件中允许哪些键(`server` 和`port`)，以及端口字段的特征是什么。

现在让我们创建一个配置文件:

```
# config.yamlserver:
  port: 6000
```

或者在 JSON 中:

```
# config.json{
  "server": {
    "port": 6000
  }
}
```

用其中一个运行我们的脚本:

```
$ python script.py -c config.yaml
 * Serving Flask app "x" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on [http://127.0.0.1:6000/](http://127.0.0.1:5000/) (Press CTRL+C to quit)
```

成功！您甚至可以提供多个配置文件来覆盖某些配置密钥，或者提供一个配置文件文件夹来加载所有配置文件。

现在假设我们在配置文件中犯了一些错误:

```
# config_with_warnings.yamlserver:
  port: 500.0
  not_needed_key: some random value
```

我们可以使用[py configuration](https://pyfiguration.readthedocs.io/en/latest/)命令行工具来找到这些:

```
$ pyfiguration inspect config -s script.py -c config_with_warnings.yaml
--------
 Errors 
--------
   ✗ Value '500.0' is not of the correct type. The allowed data type is: int
----------
 Warnings 
----------
   ! Key 'server.not_needed_key' doesn't exist in the definition and is not used in the module.
```

请注意，错误的服务器端口被标记为错误，代码未使用的密钥被标记为警告。代码没有使用的键没有错，但是它可能会导致脚本的意外结果。

# …以及更多！

配置继承，检查，解析，…这个帖子讲的太多了。前往 [GitHub 库](https://github.com/gijswobben/pyfiguration)、 [PyPi](https://pypi.org/project/pyfiguration/) 或 [ReadTheDocs](https://pyfiguration.readthedocs.io/en/latest/) 获取更多信息！