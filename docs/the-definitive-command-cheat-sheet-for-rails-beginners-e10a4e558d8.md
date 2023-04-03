# Rails 初学者的权威命令备忘单

> 原文：<https://medium.com/analytics-vidhya/the-definitive-command-cheat-sheet-for-rails-beginners-e10a4e558d8?source=collection_archive---------26----------------------->

当我还是一个 Rails 新手的时候，头几周我读完了所有的文章，理解了我正在做的事情的 30%。这是大量的信息，最让我恼火的事情之一是，经过大量努力，我终于意识到我必须做什么，我必须浏览所有的阅读材料，并找到我需要在我的终端上写下的正确命令(经过这么长时间，现在我几乎可以闭着眼睛键入它们，但起初这非常困难)。

![](img/21e708a59fd57ca7822de471aa10ecb2.png)

所以在这篇文章中，我将写下我刚开始学习时最想拥有的东西:我在一个地方学习 Rails 时使用的所有最流行的命令。我知道它们不是全部，但我希望它们是您在开始时将使用的大多数命令。

![](img/14b369ba78e0af271b947bb05c44efb9.png)

# 创建新的 rails 应用程序时:

**$rails 新名称:**非常简单，创建一个新的 rails 应用程序，然后给它命名。

**$bundle install:** 安装 Gemfile.lock 中指定的 gem 版本，如果有不兼容的版本它会报错。

**$捆绑安装——不生产:**排除生产组中的宝石。

**$bundle update:** 将您所有的 gem 依赖项更新到最新版本。

# 环境和网络浏览器:

**$rails server 或 rails s:** 如果您想通过 web 浏览器访问您的应用程序，请使用它。如果是本地的，方向通常是: [http://localhost:3000](http://localhost:3000)

**$rails 控制台或 rails c** :允许您在开发环境中从命令行与 rails 应用程序进行交互。

**$rails 控制台—沙箱:**当您希望在不更改任何数据的情况下测试一些代码时使用它，当您退出时，任何更改都会消失。

**$rails 控制台测试**:在测试环境中运行控制台。

**$重装！**:如果您更改了一些源代码，并且希望这些更改在控制台中得到反映，而无需重新启动，请使用该选项。

**$rails 服务器—环境生产**:在生产模式下运行一个 rails 应用。

# 生成:

**$rails 生成 scaffold Post name:string title:string content:text**一个 scaffold 是一组模型、数据库、控制器、视图，以及它们中每一个的测试套件。通常，您应该包括模型的名称(用单数和首字母大写)和模型的参数。在这个例子中，我们创建了一个名为 Post 的模型，带有参数 name、title 和 content。

**$rails 生成控制器帖子或者 rails g 控制器帖子:**创建一个控制器，名字应该是:首字母大写，并且用复数。

**$rails 生成控制器 post show:**如果这样做，您将拥有与上面相同的控制器，外加一个名为 show 的动作。

**$rails 生成模型 Post:** 创建一个模型，名字应该是:首字母大写，并且用单数。

**$ rails 生成模型 Post name:string title:string content:text:**相同但也包括属性:name、title 和 content。

# 迁移:

**$rails db:migrate** :运行模型及其属性的迁移。

**$ rails 生成迁移 migration_description:** 在 rails 中更改数据库模式最简单的方法就是生成迁移。不要直接对数据库进行更改。

**$ rails db:migrate:reset:**这将删除数据库信息，并在新的数据库上运行迁移。

**$rails db:seed:** 将 db/seeds.rb 文件中的数据加载到数据库中。这是用 Rails 项目所需的初始数据填充数据库的一种非常有用的方式。

# 当事情变得一团糟时:

**$ rails destroy model Post:**几乎所有用 generate 命令创建的东西都可以用 destroy 命令销毁。在这个例子中，我销毁了一个名为 Post 的模型。

**$rails db:rollback:** 这将撤销上一次迁移，然后您可以编辑该文件，并再次运行 rails db:migrate。

**$ rails db:migrate VERSION = 0:**使用它将所有迁移回滚到(包括)目标迁移。在这种情况下，我们使用版本号 0。

# 测试:

**$rails test 或 rails t** :运行我们的测试套件来验证我们的测试是否通过。

**$rails 生成 Integration _ test site _ layout**:集成测试是用来测试你的应用的各个部分是如何交互的。在这里，我们将在“test/integration”文件夹中创建一个名为“site_layout.test.rb”的集成测试。

**$rails 测试:集成:**只运行特定部分的测试，在这种情况下，它将只运行集成测试。

# 路线:

**$rails routes:** 获取应用程序中可用路线的完整列表。

# 奖励:与 Heroku 一起部署

如果您部署到 Heroku 并使用 Github，这是一个简单的分步命令行。您需要记住，您的应用程序应该位于 Github 存储库的主分支中，以便 git push heroku 命令工作。如果没有，需要使用命令:$ git push heroku your branch:master。

**$ git 状态**

**$ git add -A**

**$ git commit -m“提交描述”**

**$ git 推送**

**$ rails 测试**

**$ git 推 heroku**

**$ heroku pg:重置数据库**

**$ heroku 运行 rails db:migrate**

**$ heroku run rails db:seed**

**$ heroku 开启**

![](img/c9854b311a0b99e639116d8a98638034.png)

希望你喜欢它！快乐编码