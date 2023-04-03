# 在 Python 中用 Pytest 测试时伪造 Redis/PostgreSQL 连接

> 原文：<https://medium.com/analytics-vidhya/faking-redis-postgresql-connections-while-testing-with-pytest-in-python-727d9b9660f7?source=collection_archive---------10----------------------->

我最近不得不为使用 Redis 和 PostgreSQL 数据库存储数据的函数编写一些 Python 单元测试。为了实现这一点，我必须学习如何使用 Pytest 和能够伪造数据库连接的包。由于缺乏良好的文档和有用的在线示例，后者被证明是特别具有挑战性的。我花了很多时间谷歌，弄清楚如何在测试中建立这些虚假的联系，并建立了一个测试框架。因此，我想分享我所做的，来帮助其他想写类似测试的人节省时间。我将首先做一个假 Redis 连接的例子，然后看一个假 PostgreSQL 连接的例子。我将假设对 Pytest(我是从这篇有用的文章[https://medium . com/test cult/intro-to-test-framework-Pytest-5b 1 ce 4d 011 AE](/testcult/intro-to-test-framework-pytest-5b1ce4d011ae))和 Redis/PostgreSQL 数据库有所了解。关于包括测试文件在内的所有代码，请参见[https://github.com/paluchasz/Pytest-testing](https://github.com/paluchasz/Pytest-testing)。

# Redis 示例

对于 Redis 示例，假设我们想要测试一个对某些 Redis 键的值求和的函数。这种函数的一个简单例子是

```
def sum_redis_key_values(rc, key_format):
    total = 0
    for redis_key in rc.scan_iter(key_format.format(i="*")):
        total += int(rc.get(redis_key))
    return total
```

在这个例子中，我选择的键是: *key:0，key:1，key:2，key:3* (so `key_format = “key:{i}"`)，它们的值是整数。

假设我们想要测试三个不同测试用例的功能，并给出每个测试用例的通过/失败。这可以通过使用 Pytest 的 parametrize decorator 和一个返回元组列表的函数轻松实现。例如，将下面的代码放在一个文件中(测试 Python 的 sum()方法)并运行*$ Python 3-m pytest filename . py-v*

```
import pytestdef get_sum_test_data():
    return [([3, -2, 6, -8], -1), ([20, 0, -30, 40], 30), ([2, 4, 1, 5], 12)]

@pytest.mark.parametrize('nums, result', get_sum_test_data())
def test_sum(nums, result):
    assert sum(nums) == result
```

将产生三个 PASS 语句，每个测试用例一个。将数字 30 改为其他数字将导致通过/失败/通过。

为了真正伪造一个 Redis 连接，我使用了 [*birdisle*](https://pypi.org/project/birdisle/) 包( *02/2021 更新*:如果你使用的是 Mac OS，我建议使用 [*redislite*](https://pypi.org/project/redislite/) 包来代替，参见文章末尾的解释)。从文档中

> bird isle(“lib redis”的变位词)是 redis 的一个修改版本，它作为一个库在另一个进程中运行。主要目的是简化单元测试，提供一种在 redis 服务器上运行测试的方法，而不需要启动一个单独的进程并确保它被正确拆除。”

要使用它，可以简单地做以下事情

```
from birdisle import redisrc = redis.StrictRedis(decode_responses=True)
```

通常情况下，如果要连接到本地 Redis 数据库，您需要传入`host = "localhost”`和`port = 6379`。如果您将这个连接对象`rc`与您从标准 Redis 包中获得的对象进行比较，您可以看到这个对象与 *LocalSocketConnection* 略有不同。现在，您可以使用标准 *redis* 包中的任何 Redis 命令。我喜欢保存一个全局`rc` (redis client)变量，这样就可以在文件中的任何地方轻松地访问连接。

现在我们需要另外两个函数。在每个测试用例之后，一个创建数据，一个清除假 Redis 连接中的数据。您可以自己检查新的键实际上没有添加到本地 Redis 中。下面是完整的测试文件。

使用假 Redis 连接的完整测试文件

让我们总结一下它的作用。当您运行*$ python 3-m pytest test _ 作伪 _redis.py -v* 时:

*   Pytest 首先识别文件中哪些函数的名称以单词“test”开头或结尾。在我们的例子中，我们只有一个这样的函数叫做`test_sum_keys()`。
*   然后从 Pytest 的参数化装饰器调用`load_data_and_connect()`函数。
*   在它内部，如果一个 Redis 连接还不存在的话，就会建立一个假的 Redis 连接。
*   所有数据都被加载并转换成一个元组列表，如前面显示的 Pytest 示例所示。(因为我只使用了三个测试用例，所以我将每个测试用例存储在一个单独的 json 文件中，但是可能有更好的方法来做到这一点)。
*   一旦这个函数被执行，Pytest 就依次运行每个元组的测试函数，作为每个测试用例的输入。
*   对于每个测试用例，创建假 Redis 数据，执行我们正在测试的函数`sum_redis_key_values()`，清除 Redis 数据，并断言总和是否等于预期结果。
*   Pytest 记录断言的通过/失败，并继续下一个测试用例。

# PostgreSQL 示例

伪造 PostgreSQL 连接比预期的更具挑战性。首先，提醒一下如何在 Python 中连接和使用 psql。我使用的包是 *PyGreSQL* 。要建立连接，我们可以执行以下操作

```
import pgdbCONNECTION = pgdb.connect(user=info['user'], host=info['host'], database=info['database'], port=info['port'])
```

其中`info` 是要指定的字典，可以从配置文件中加载。(注意，我认为还有另一个 pgdb Python 包，因此您需要安装正确的包，否则您可能会得到类似“pgdb 没有连接方法”的错误)

为了查询数据库，我们可以这样做

```
cursor = CONNECTION.cursor()    
records = cursor.execute("""SELECT age FROM students""").fetchall()
cursor.close()
```

现在，为了创建一个假的 psql 连接，我使用了 *testing.postgresql* 包([https://pypi.org/project/testing.postgresql/](https://pypi.org/project/testing.postgresql/))。注意，这个包需要安装一个本地 PostgresSQL 数据库(与 birdisle 包相反，我认为它不需要 Redis 数据库)。该包提供了一个假的数据库/用户/主机/端口，可用于与 PyGreSQL 包建立假连接，如下所示

```
import pgdb
import testing.postgresqlpsql = testing.postgresql.Postgresql()
info = psql.dsn()
CONNECTION = pgdb.connect(user=info['user'], host=info['host'], database=info['database'], port=info['port'])
```

例如，假设我们要测试一个函数，该函数对 psql 中“学生”关系(表)中所有学生的年龄求和。该函数可能类似于

```
def sum_ages():
    records = query_database(operation="""SELECT age FROM students""")
    ages = [r.age for r in records]
    return sum(ages)
```

其中`query_database()`是另一个函数，它执行前面显示的所有光标操作。

为了在伪 psql 数据库中创建学生关系, *Pandas* 包非常有用。它有一个方便的`to_sql()`方法，可以将 Pandas 数据帧转换成 psql 关系。它需要传递一个引擎对象，告诉 Pandas 要插入哪个 psql 数据库。为了获得这个对象，我们需要使用另一个名为 *sqlalchemy* ***的包。*** 将制作学生表格所需的数据保存为 csv 文件是有意义的，这样就可以用`read_csv()`方法轻松地将其加载到 Pandas dataframe 中。把这些放在一起，我们需要做一些事情

```
import testing.postgresql
import pandas as pd
from sqlalchemy import create_enginepsql = testing.postgresql.Postgresql()
engine = create_engine(psql.url())students_df = pd.read_csv(file)
students_df.to_sql('students', engine, if_exists='replace')
```

`if_exists = “replace"`告诉熊猫替换同名的现有关系。还有其他有用的东西可以传递给`to_sql()` 方法，比如`dtype` ，它指定列的数据类型。

在处理 psql 连接时，我们需要在每个测试用例之后回滚连接(回滚用于撤销事务，似乎比试图清除数据库更容易——我也遇到过一些没有这样做的问题),并在最后一个测试用例之后关闭连接。因此，我们还需要跟踪总的测试用例以及测试用例 id。下面是完整的测试文件

使用假 PostgreSQL 连接的完整测试文件

注意，我们需要做的事情与 Redis 示例略有不同，并在每个测试函数中伪造连接，而不是在 parametrize decorator 中调用的`load()`函数。旧的方法在这里仍然有效，但是如果我们在这个文件中有多个使用相同参数化的测试函数，问题就出现了。如果有多个函数，Pytest 首先定位所有函数，并在运行所有测试之前依次执行每个参数化装饰器。因此，每次调用 load 函数时，我们的连接对象都会被覆盖。(我是在一次运行多个测试文件的 Pytest 时发现这个问题的；他们是单独经过，而不是一起经过！)

# 排除故障

Pytest 中有一些使用`set_trace()`方法的调试([https://qxf 2 . com/blog/debugging-in-python-using-py test-set _ trace/](https://qxf2.com/blog/debugging-in-python-using-pytest-set_trace/))但是我没有发现它特别有用。但是，如果您使用 PyCharm，您可以设置一个非常有用的运行/调试 Pytest 配置，让您运行 Pytest 并逐行检查代码。

# 制作测试框架

首先，我将所有伪造的数据库连接功能转移到一个单独的共享文件中。然后我制作了一个类装饰器，它将处理每个测试的所有连接，以避免测试函数中的重复代码。由于装饰器必须包装测试函数，assert 语句成了一个问题，因为如果断言失败，就不会执行其他代码，因此装饰器中的连接不会关闭。幸运的是，Pytest 有一个额外的包叫做 *pytest_check* ，可以如下使用

```
import pytest_check as checkcheck.equal(a, b)
```

这相当于断言 a 等于 b，但它允许程序继续运行。因此，您甚至可以在每个测试函数中进行多次检查。

# 更新

**02/2021:** 后来我发现用来伪造 Redis 连接的 *birdisle* 包有一个很大的缺点。如果您在 Mac OS 上工作，它将无法安装，测试也无法运行。一个痛苦的解决方法是拥有一个 Linux 虚拟机，并跨其同步您的文件。不过我后来发现有一个替代包叫 [*redislite*](https://pypi.org/project/redislite/) 也一样好用。如果你想伪造一个连接，你只需要做

```
from redislite import StrictRedisrc = StrictRedis(decode_responses=True)
```

并且应该支持 *redis* 包的所有命令。

# 摘要

我希望这篇文章对您有用，如果您发现自己处于相同的位置，并且需要使用相同的软件包，它将为您节省一些时间。如果任何人对可以做得更好/更容易有任何问题或建议，请留下评论。