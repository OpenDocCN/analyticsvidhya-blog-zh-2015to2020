# 使用 SQLAlchemy 的对象关系映射

> 原文：<https://medium.com/analytics-vidhya/object-relational-mapping-with-sqlalchemy-233da7c0f064?source=collection_archive---------7----------------------->

![](img/e401bdbc7c6275feb659e9282738de7e.png)

# 什么是对象关系映射？

对象关系映射(ORM)是一种用于将对象映射到数据库的技术。使用 ORM，我们可以直接创建包含类的表和包含创建的对象(类的实例)的表中的数据。这种编程技术提供了一个标准接口，允许开发人员创建可用于与各种数据库引擎通信的代码，而不是为每个数据库引擎进行定制。

# SQLAlchemy

SQLAlchemhy 是 Python 中的 ORM 实现之一，它是一个方便 Python 程序和各种数据库引擎之间通信的库，本文将使用它来持久化和查询数据。对我来说，使用 ORM 的两个主要优点是，它允许我完全不用在代码中编写 sql 查询(不是因为我讨厌 sql ),它还确保我可以编写一个通用代码，可以用于各种关系数据库引擎(如果我们需要更改数据库引擎，只需很少或不需要调整)。确保这种与多个数据库引擎协同工作的能力的一些特性将在下面几节中讨论。

## SQLAlchemy 数据类型

SQLAlchemy 支持隐式和显式声明列的数据类型。它允许将 Python 数据类型映射到关系数据库列中最常见的数据类型，这样 SQLAlchemy 将选择目标数据库中可用的最佳数据库列类型，并映射到特定的 Python 数据类型，如下所示；

*   Integer() — INT
*   String() — ASCII 字符串— VARCHAR
*   Unicode()-Unicode 字符串-VARCHAR 或 NVARCHAR，具体取决于数据库
*   Boolean() — BOOLEAN、INT、TINYINT 取决于数据库对布尔类型的支持
*   DateTime()-DateTime 或 TIMESTAMP 返回 Python datetime()对象。
*   Float() —浮点值
*   numeric()-使用 Python Decimal()的精确数字

此外，它允许通过使用 SQL 标准语法或数据类型的数据库后端子集通用的语法来显式声明数据类型。然后，它还提供了一种指定特定于供应商的类型的方法，比如 MySQL 的 BIGINT 和 PostgreSQL 的 INET。

## SQLAlchemy 引擎

SQLAlchemy 引擎是任何 SQLAlchemy 应用程序的起点，因为每当我们想要使用 SQLAlchemy 与数据库进行交互时，我们都需要创建一个引擎。通过运行从 sqlalchemy 导入的 create_engine 函数来创建引擎。该函数使用数据库 URI(方言+驱动://用户名:密码@主机:端口/数据库)来创建引擎对象。引擎一旦创建，就可以用来直接与数据库交互，或者传递给一个会话对象来使用 ORM。在本文中，我们将把引擎传递给会话对象，因为我们正在讨论 ORM。注意，create_engine 函数不直接建立任何实际的 DBAPI 连接。引擎引用连接池，这意味着在正常情况下，当引擎对象仍然驻留在内存中时，存在打开的数据库连接。连接池是在内存中维护活动数据库连接池的一种方式。因此，为了实际连接到数据库，需要调用引擎对象的 connect 方法或依赖于该方法的操作。这种方法导致使用池中的一个数据库连接来连接数据库。

## SQLAlchemy 连接池

连接池是对象池设计模式的一种实现。对象池是缓存对象以便重用的一种方式。对象池设计模式提供了显著的性能提升，尤其是在初始化类实例的成本很高的情况下。此外，当池为空且向池发送对象请求时，池可以通过创建新对象来自动增长，或者池可以受创建的对象数量的限制。因此，连接池是一种标准技术，用于在内存中维护长时间运行的连接以实现高效重用，并为应用程序可能同时使用的连接总数提供管理。SQLAlchemy 实现了各种连接池模式。在大多数情况下，create_engine 函数返回的引擎具有一个队列池，该队列池具有合理的池默认值。

## SQLAlchemy 方言

SQLAlchemy 使用 SQLAlchemy 方言与 Python 数据库 API 规范(DBAPI)的各种实现进行通信。DBAPI 定义了一个标准接口来访问用 Python 开发的模块/库的数据库。SQLAlchemy 方言描述了如何与特定类型的数据库/DBAPI 组合进行对话。DBAPI 的一些实现分别是 pyscopg2 和 mysql-connector-python for PostgreSQL 和 MySQL。注意:所有方言都要求安装适当的 DBAPI。

现成的、受支持的 SQLAlchemy 方言有；

*   一种数据库系统
*   关系型数据库
*   SQLite
*   神谕
*   Microsoft SQL Server

注意:SQLAlchemy 方言比较多。

## SQLAlchemy 基本关系模式

我们现在可以使用我们到目前为止学到的所有知识来将类之间的关系映射到表之间的关系。SQLAlchemy 支持四种类型的关系:一对一、一对多、多对一和多对多。接下来将实现这些关系。但是在此之前，我们需要安装一些库并运行一个实际的数据库，以便在 SQLAlchemy 上查询数据。众所周知，SQLAlchemy 支持许多不同的数据库引擎，但是我们将使用 PostgreSQL。但是首先，让我们为我们的项目创建一个新的目录和一个虚拟环境，如下所示；

```
>> mkdir object-relational-mapping 
>> cd object-relational-mapping 
>> python3 –m venv orm-venv 
>> venv/Scripts/activate
```

上面的第一个和第二个命令创建项目目录(对象关系映射)并分别改变目录，而第三个和第四个命令创建虚拟环境并分别激活虚拟环境。

## 安装 SQLAlchemy 及其依赖项

首先，请记住，所有方言都要求安装适当的 DBAPI，因为我们将使用 PostgreSQL 数据库引擎，所以我们需要安装 psycopg2，这是连接到 PostgreSQL 引擎的最常见的 DBAPI 实现。另外，SQLAlchemy 不是 Python 标准库的一部分，所以我们也需要安装它。实际上只需要这两个库，因此要将这些库安装到我们的虚拟环境中，我们将使用 pip，如下所示:

```
>> pip install sqlalchemy psycopg2
```

## 启动 PostgreSQL

有几种方法可以获取 PostgreSQL 数据库引擎的实例。在本教程中，我们将利用亚马逊网络服务(AWS)上的 RDS 服务。另一种选择是在我们当前的环境中本地安装 PostgreSQL。因此，为了在 AWS 上创建 PostgreSQL 的实例，如果您还没有帐户，那么您需要在 AWS 上创建一个帐户。注意:对于这个例子，我们将保持在 AWS 的自由层限制内。因此，一旦我们登录到我们的 AWS 帐户，单击导航栏中的服务，然后单击数据库下的 RDS。在打开的新页面上，单击“创建数据库”。然后按照以下步骤创建 DB 实例；

*   填写以下字段；数据库实例标识符、主用户名、主口令和确认口令
*   接下来，向下滚动到“Additional connectivity configuration ”,单击箭头显示该类别下的设置，然后单击“Publicly accessible”下的“Yes ”,以便我们可以从外部连接到数据库。
*   此外，向下滚动到“附加配置”并单击箭头以显示此类别下的所有设置，然后填写“初始数据库名称”字段
*   最后，单击 create database，等待几分钟，直到数据库的状态变为“可用”。
*   一旦数据库变得可用，您可以单击它来查找将用于连接到数据库的“端点”。单击新创建的数据库后，该端点位于“Connectivity & Security”选项卡下。

注意:为了不产生任何费用，请记住在示例后删除数据库。

## 映射类以创建关系

在这里，我们将实现一对多关系，并且将向任何希望这样做的人介绍如何扩展该示例以实现其他关系的想法。这个例子是基于创建两个数据库表；俱乐部和球员。俱乐部表将由足球俱乐部组成，而球员表将包含足球运动员，并且关系是这样的，足球运动员只能为俱乐部效力，而俱乐部可以由几名球员组成。首先，让我们创建一个基础文件来创建我们的引擎，然后创建一个会话来使用 ORM。此外，我们将在基本文件中定义一个基类，我们的 Python 类将从该基类继承，以便生成适当的数据库表。基础文件的内容如下所示:

## base.py

```
from sqlalchemy import create_engine 
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy.orm import sessionmaker #creating our engine 
engine = create_engine('postgresql+psycopg2://yinka:password@orm.cwmprma8kzpp.us-east-1.rds.amazonaws.com:5432/orm') #remember that the engine needs to be passed to a Session object in order to be able to work with the ORM 
Session = sessionmaker(bind=engine) #the base class for defining our classes in order to produce the appropriate Tables 
Base = declarative_base()
```

请注意，数据库 uri 的一般格式是:

```
>> dialect+driver://username:password@host:port/database
```

然后根据凭据，我们在 AWS 中创建数据库时使用“主用户名”、“主密码”和“初始数据库名称”，以及创建的数据库的“端点”；

```
>> postgresql+psycopg2://Master username:Master password@endpoint:5432/Initial database name
```

因此，请编辑代码中的数据库 uri，以便能够连接到您的数据库，因为当您看到此信息时，我已经删除了该数据库。谢谢:)

## club.py

接下来，让我们创建俱乐部表，如下所示；

```
from sqlalchemy import Column, String, Integer, Date 
from base import Base class Club(Base): 
    __tablename__ = ‘clubs’     id = Column(Integer, primary_key=True) 
    club_name = Column(String) 
    club_stadium = Column(String) 
    date_founded = Column(Date)     def __init__(self, club_name, club_stadium, year_founded):
        self.club_name = club_name 
        self.club_stadium = club_stadium 
        self.year_founded = year_founded
```

这是从基类继承来创建俱乐部表的。该表有四列；

*   主键
*   俱乐部的名称
*   俱乐部体育场的名称
*   俱乐部成立的日期

注意，我们使用 Python 数据类型(整数、字符串、日期等)，SQLAlchemy 根据目标数据库引擎将这些数据类型映射到最合适的数据库列类型。

player.py

接下来，让我们如下创建球员表；

```
from sqlalchemy import Column, String, Integer, Date, Table, ForeignKey 
from sqlalchemy.orm import relationship 
from base import Base class Player(Base): 
    __tablename__ = 'players' 

    id = Column(Integer, primary_key=True) 
    player_name = Column(String) 
    player_number = Column(Integer) 
    club_id = Column(Integer, ForeignKey('clubs.id')) 
    club = relationship('Club', backref='players')     def __init__(self, player_name, player_number, club):
        self.player_name = player_name 
        self.player_number = player_number 
        self.club = club
```

玩家表也有四列；

*   主键
*   玩家的名字
*   球员的球衣号码
*   球员效力的俱乐部的信息是从俱乐部表继承的。

“俱乐部”变量不是一个列，而是定义球员和俱乐部表之间的关系。

## update.py

是时候实际创建我们的数据库模式并向其中插入一些数据了。这是按如下方式完成的:

```
from player import Player 
from base import Session, engine, Base 
from club import Club 
from datetime import date #create database schema 
Base.metadata.create_all(engine) #create a new session 
session = Session() #create clubs 
manchester_united = Club('Manchester United', 'Old Trafford', date(1878, 1, 1)) 
chelsea = Club('Chelsea', 'Stamford Bridge', date(1905, 3, 10))
juventus = Club('Juventus', 'Allianz Stadium', date(1897, 11, 1)) #create players de_gea = Player('David de Gea', 1, manchester_united) 
pogba = Player('Paul Pogba', 6, manchester_united) 
kante = Player("N'Golo Kante", 7, chelsea) 
ronaldo = Player('Cristiano Ronaldo dos Santos', 7, juventus) #persist the data 
session.add(manchester_united) 
session.add(chelsea) 
session.add(juventus) session.add(de_gea) 
session.add(pogba) 
session.add(kante) 
session.add(ronaldo) #commit and close session 
session.commit() 
session.close()
```

首先，我们创建了数据库模式，然后创建了一个用于持久存储数据的会话。接下来，我们创建了一些俱乐部和球员对象，它们将分别被转换成俱乐部和球员表中的数据行。然后，我们持久化数据，并最终在关闭会话之前提交数据。这样，数据现在就在我们的数据库中了。

## query.py

最后，让我们查询我们的数据库，以确认我们在上一节中持久存储的数据实际上已经被保存。这是通过下面的代码实现的；

```
from club import Club 
from base import Session 
from player import Player #creates a session 
session = Session() #extracts all the players 
players = session.query(Player).all() for player in players: 
    print(f’{player.player_name} plays for {player.club.club_name} and wears shirt number {player.player_number}’)
```

它的输出如下:

```
David de Gea plays for Manchester United and wears shirt number 1 Paul Pogba plays for Manchester United and wears shirt number 6 N'Golo Kante plays for Chelsea and wears shirt number 7 Cristiano Ronaldo dos Santos plays for Juventus and wears shirt number 7
```

太棒了。！！

## 最后

感谢您的光临，希望这对您有所帮助。正如所承诺的，为了完成这个例子来实现其他的关系；

一对一:一个新的表格，包含俱乐部历史上进球最多的球员的信息。由于一个俱乐部只能有一个历史最高进球得分者，并且一名球员在多个俱乐部中成为历史最高进球得分者的情况非常罕见，因此可以在俱乐部表中建立一对一的关系。

多对多:包含赞助商信息的新表。由于一个俱乐部可以有多个赞助商，而一个赞助商也可以赞助多个俱乐部，因此可以使用俱乐部表建立多对多关系。