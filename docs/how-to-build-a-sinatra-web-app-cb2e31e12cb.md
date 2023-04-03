# 如何构建 Sinatra 网络应用程序

> 原文：<https://medium.com/analytics-vidhya/how-to-build-a-sinatra-web-app-cb2e31e12cb?source=collection_archive---------18----------------------->

这是对 Sinatra 和 ActiveRecord 的介绍，以及我用来创建具有模型、视图、控制器(MVC)结构和数据持久性的全栈应用程序的一些方法。我目前正在开发一个应用程序，允许用户使用安全密码注册，并创建、读取、更新和删除订单。我在肉类市场的全食超市工作，每年的这个时候都会有成百上千的顾客订单，仅仅用纸和笔很难记录下来。我正在开发这个应用程序，希望有一天我能在工作中使用它来跟踪这些订单。

# 那么从哪里开始呢？

在开始之前，您不需要确切地知道所有的代码，但是最好知道您希望成品看起来是什么样的，您希望它如何表现，以及诸如“我将有多少个模型？”这样的问题的答案"我需要使用什么数据类型？"在你继续和开始之前，这应该是清楚的。我的文件结构如下所示:

```
├── Gemfile
    ├── Gemfile.lock
    ├── README.md
    ├── Rakefile
    ├── app
    │ ├── controllers
    │ │ ├── application_controller.rb
    │ │ ├── users_controller.rb
    │ │ ├── orders_controller.rb
    │ │
    │ ├── models
    │ │ ├── user.rb
    │ │ ├── order.rb
    │ │ 
    │ |── views
    │ ├── user
    │ │ ├── signup.erb
    │ │ ├── login.erb
    │ │ 
    │ ├── orders
    │    ├── show.erb
    │    ├── new.erb
    │    |── edit.erb
    │  
    │ 
    ├── config
    │ └── environment.rb
    ├── config.ru
    ├── db
    ├── public
```

一旦我建立了基本的结构，我喜欢提交并推到 GitHub。您应该在实际编码时间的每 7-10 分钟提交一次描述性的提交消息。

# 添加您的宝石

你可以随时添加更多的宝石，但在开始时，你需要添加一切将用于启动你的应用程序。这是我项目的 gem 文件:

```
source "https://rubygems.org"gem 'sinatra'
gem 'activerecord', :require => 'active_record'
gem 'sinatra-activerecord', :require => 'sinatra/activerecord'
gem 'rake'
gem 'require_all'
gem 'sqlite3'
gem 'thin'
gem 'shotgun'
gem 'pry'
gem 'bcrypt'
gem "tux"group :test do
  gem 'rspec'
  gem 'capybara'
  gem 'rack-test'
  gem 'database_cleaner', git: 'https://github.com/bmabey/database_cleaner.git'
end
```

# 哦，这么重要的耙子

Rakefile 将需要“sinatra/activerecord/rake”并加载您的环境。我在自己的任务中添加了一个任务，启动一个“窥探”控制台来测试我的代码:

```
require_relative './config/environment'
require 'sinatra/activerecord/rake'task :console do
    Pry.start
end
```

# 配置文件夹和 environment.rb

这可以被视为项目中最重要的文件之一。它是您描述所有其他依赖项的地方，也将用于装载您的数据库适配器。数据库适配器是 ActiveRecord 知道在哪里存储数据的方式。我正在使用 SQLite 查询语言。这是典型的环境。rb:

```
require 'bundler'
Bundler.require
ActiveRecord::Base.establish_connection(
  :adapter => 'sqlite3',
  :database => 'db/development.sqlite'
)
require_all 'app'
```

# 数据库和迁移

一旦你的适配器，rakefile 和你所有的宝石设置好了；您可以开始设置迁移和数据表。ActiveRecord 为您提供了作为版本控制的强大迁移工具。您可以使用命令`rake db:create_migration NAME="name of migration"`创建这些版本。这将在您的“db”目录中自动生成一个“migrate”文件夹，您可以在其中执行创建表等操作。下面是我用来创建用户表的迁移:

```
class CreateUsers < ActiveRecord::Migration[6.0]
  def change
    create_table :users do |t|
      t.string :username
      t.string :password_digest
    end
  end
end
```

# 配置. ru

config.ru 文件是另一个非常重要的文件。该文件将加载您的环境并运行您的控制器:

```
require './config/environment'use UserController
use OrdersController
use Rack::MethodOverride
run ApplicationController
```

# 应用控制器

应用程序控制器将继承 Sinatra，这将为它提供定义您的路线所需的所有功能。所有其他控制器都将继承这个控制器。如果您想要启用会话，它应该是这样的:

```
class ApplicationController < Sinatra::Base configure do
    set :public_folder, 'public'
    set :views, 'app/views'
    enable :sessions
    set :session_secret, "session_secret"
  end get '/' do 
    "Hello, World!"
  end end
```

# 模型

模型是继承自 ActiveRecord 的 ruby 类，active record 将为它们提供对应用程序非常重要的所有关联所需的功能。这些有价值且近乎神奇的关联是通过使用像`has_many`和`belongs_to`这样的抽象来实现的。这里的另一个重要成分是`has_secure_password`，它将通过“bcrypt”宝石给这个类一个安全的密码:

```
class User < ActiveRecord::Basehas_many :orders
has_secure_passwordendclass Order < ActiveRecord::Basebelongs_to :userend
```

# 控制器、路线和视图

这就是 Sinatra 应用程序的基本结构。此时，您已经准备好构建您需要的任何路线和视图。

*原载于 2019 年 12 月 3 日 https://jcguest.github.io。*