# 使用 Faker 和工厂男孩测试 Django 的模型

> 原文：<https://medium.com/analytics-vidhya/factoryboy-usage-cd0398fd11d2?source=collection_archive---------0----------------------->

![](img/7b7a34e913b56ea4bb5918d38d7d79d3.png)

**factory_boy** 库(模仿 Rails 中的 Factory Girl)为测试生成数据。在 Django 中，测试数据被称为 fixtures，从代码文件中加载。

我们可以安装这个库:

```
$ pipenv install factory_boy
```

然后，我将在我的应用程序中创建一个这样的结构:

```
$ core/
  ----tests/
  ---------/__init__.py
  ---------factories/
  ------------------/__init__.py
  ------------------/company.py
  ---------/test_models/
  --------------------/__init__.py
  --------------------/test_company.py
```

**工厂文件夹**包含将在测试阶段创建公司对象的工厂公司。而 **test_company.py** 文件包含**公司**的测试代码。

# 使用工厂

*   通过 ORM 创建一个对象。
*   使用**元类**来告诉你的工厂使用哪个模型。

我们将在`**core/factors/company . py**中创建公司工厂:

```
import factoryclass CompanyFactory(factory.DjangoModelFactory):
    """
       | id  | name      |
       |:----|:----------|
       |  1  | X Company |
    """ class Meta:
        model = 'core.Company'
        django_get_or_create = ('name',) name = 'company name'
```

工厂的 Meta 设置了 **django_get_or_create** ，这意味着工厂将调用 django 内置的`**company . objects . get _ or _ create**方法。

现在，只要调用` **CompanyFactory.build()** `就会生成一个填充的 **company** 实例:

```
# Build a Company instance and override company
>>> company_obj = CompanyFactory.build(name='X Company')
>>> company_obj.name
    "X Company"
```

现在，在你的数据库中创建并拥有一家公司就像写下`**company factory . create()**一样简单:

```
# Returns a saved Company instance
>>> company = Company.create(name='X Company')# or# Same as CompanyFactory.create()
>>> company = CompanyFactory(name='X Company')
```

您可以在测试中访问该对象:

```
from test_factories import CompanyFactory
company = CompanyFactory()
```

使用工厂时，可以在中传递不同的值:

```
company = CompanyFactory(name='Y Company')
```

我们公司实例的名称将是` **Y Company** `,而不是` **company name** `。

我们想创造更多的物体:

```
>>> companies = Company.create_batch(3, name=factory.Sequence(lambda n: 'Name {0}'.format(n)))>>> len(companies)
    3>>> [company.name for company in companies]
    ["Name 1", "Name 2", "Name 3"]
```

工厂是用于为特定模型生成数据的类。这些被称为测试。

这是我们更新的“**main/tests/test _ models/company . py**”:

```
from django.test import TestCasefrom ..factories import CompanyFactoryclass CompanyModelsTestCase(TestCase):
    [@classmethod](http://twitter.com/classmethod)
    def setUpTestData(cls):
        cls.Company = CompanyFactory._meta.model def test_crud_company(self):        
        # create
        company = CompanyFactory(name='X Company') # read
        self.assertEqual(company.name, 'X Company')
        self.assertEqual(company.pk, company.id)
        self.assertQuerysetEqual(
            self.Company.objects.all(),                 
            ['<Company: X Company>']
        ) # update
        company.name = 'Y Company'
        company.save()
        self.assertQuerysetEqual(
            self.Company.objects.all(),  
            ['<Company: Y Company>']
        )
```

# 工厂男孩的特征:

*   使用一个使用 lambda 函数的` **Sequence** `对象来动态创建唯一的字段值。比如说；

```
name = factory.Sequence(lambda n: ‘company_%d’ % n)`
```

*   使用“ **LazyAttribute** ”对象返回已定义字段的值。比如说；

```
slug = factory.LazyAttribute(lambda o: o.name)
```

*   子工厂
*   多对多关系

## 顺序

如果你想创建一个给定对象的多个实例和唯一属性，
你可以添加**工厂。顺序**。

```
import factoryclass CompanyFactory(factory.DjangoModelFactory):
    """
       | id  | name      |
       |:----|:----------|
       |  1  | X Company |
    """ class Meta:
        model = 'core.Company'
        django_get_or_create = ('name',) name = factory.Sequence(lambda n: 'Company %d' % n)
```

当这个工厂被调用时，每次创建一个唯一的公司， **n** 自动增加。

## 惰性属性

` **o** `是工厂内正在构建的实例。
你可以使用字符串格式，加上一些属性。

```
import factoryclass CompanyFactory(factory.DjangoModelFactory):
    """
       | id  | name      |
       |:----|:----------|
       |  1  | X Company |
    """ class Meta:
        model = 'core.Company'
        django_get_or_create = ('name',) name = factory.Sequence(lambda n: 'Company %d' % n)
    slug = factory.LazyAttribute(lambda o: o.name)
```

## 子工厂

我需要在 Django 应用程序中定义一个测试工厂。模型之间是一对多的关系，例如**公司**有很多**签约** **公司**。工厂男孩允许你在你的工厂里使用分工厂。**子工厂**用于 FK 字段。

```
import factoryclass CompanyFactory(factory.DjangoModelFactory):
    """
       | id  | name      |
       |:----|:----------|
       |  1  | X Company |
    """ class Meta:
        model = 'core.Company'
        django_get_or_create = ('name',) name = factory.Sequence(lambda n: 'Company %d' % n)
    slug = factory.LazyAttribute(lambda o: o.name)class ContractedCompanyFactory(factory.DjangoModelFactory):
    """
        | id  | start_date      | end_date      | company |
        |:----|:--------------------------------|:--------|
        |  1  | 2019-05-13      | 2020-05-13     | 1      |
    """ class Meta:
        model = 'core.ContractedCompany' start_date = parse_date('2019-05-13')
    end_date = parse_date('2020-05-13')
    company = factory.SubFactory(CompanyFactory)
```

我将在我的测试中创建一个签约公司:

```
contracted_company = ContractedCompanyFactory()
```

## 多对多关系

我们为此功能使用的基础构建模块是`post_generation`:

```
class CompanyFactory(factory.DjangoModelFactory):
    """
       | id  | name      |
       |:----|:----------|
       |  1  | X Company |
    """ class Meta:
        model = 'core.Company'
        django_get_or_create = ('name',) name = factory.Sequence(lambda n: 'Company %d' % n)
    slug = factory.LazyAttribute(lambda o: o.name) @factory.post_generation
    def phones(self, create, extracted):
        if not create:
            return
        if extracted:
            for phone in extracted:
                self.phones.add(phone) [@](http://twitter.com/factory)factory.post_generation
    def emails(self, create, extracted):
        if not create:
            return
        if extracted:
            for email in extracted:
                self.emails.add(email) [@](http://twitter.com/factory)factory.post_generation
    def addresses(self, create, extracted):
        if not create:
            return
        if extracted:
            for address in extracted:
                self.addresses.add(address)
```

呼叫`CompanyFactory()`或`CompanyFactory.build()`时，会创建**无**无**电话**或**无**无**电子邮件**或**无**无**地址**绑定。但是当`CompanyFactory.create(phones=(phone1, phone2, phone3), .....)`被调用时，`phones`声明会将传入的电话添加到公司的电话集中。

# 例子:假货和工厂

*   **骗子**:生成假数据
*   制造我们模型的工厂

在我们的`Django`应用中，我们有 MTV(模型模板视图)结构。首先，我们必须为测试创建模型。如果您想要生成随机字符串或者生成我们模型的唯一实例，我们需要使用它们。

# 启动 Django 项目

我们已经创建了一个简单的`Django`应用程序。查看我们的`models.py`文件:

```
**from** django.contrib.auth **import** get_user_model
**from** django.db **import** modelsUser **=** get_user_model() **class** **NewsContent**(models**.**Model):
    headline **=** models**.**CharField(max_length**=**255)
    body **=** models**.**TextField()
    author **=** models**.**ForeignKey(
        to**=**User,
        on_delete**=**models**.**CASCADE,
        related_name**=**'contents_by_author'
    ) **def** **__str__**(self):
        **return** self**.**headline
```

# 功能

现在，我们的系统中有了一些自定义逻辑。没错！是时候添加一些测试了。查看我们的 **views.py** 文件:

```
**from** django.http **import** HttpResponseNotAllowed
**from** django.shortcuts **import** render

**from** .models **import** NewsContent

**def** **news_content_list**(request, author_id):
    **if** request**.**method **==** 'GET':
        object_list **=** NewsContent**.**objects**.**filter(
            author__id**=**author_id
        )
        **return** render(request, 'news_content_list.html', locals())
    **return** HttpResponseNotAllowed(['GET'])
```

查看我们的 **news_content_list.html** 文件:

```
{% for news_content in object_list %}
   <ul>
        <li>
            {{ news_content.headline }}, {{ news_content.author }}
        </li>
        <li>{{ news_content.body }}</li>
    </ul>
{% endfor %}
```

# 计划我们的测试

让我们添加第一个测试。如果我们有一个作者的 5 个新闻内容，我们必须确保视图将它们全部列出。

```
*# in tests/test_views.py*
**from** django.contrib.auth **import** get_user_model
**from** django.test **import** TestCase
**from** django.urls **import** reverse

**from** .models **import** NewsContent

User **=** get_user_model()

**class** **NewsContentListViewTests**(TestCase):
    **def** **setUp**(self):
        self**.**author **=** User**.**objects**.**create_user(
            'john', 'lennon@thebeatles.com'
        )
        self**.**news_content **=** NewsContent**.**objects**.**create(
            headline**=**'Real Madrid frustrated by Athletic',
            body**=**'Real Madrid hit the woodwork three times',
            author**=**self**.**author
        )
        self.url **=** reverse(
            'news:news_content_list', 
             args**=**(self**.**author**.**id,)
        )

    **def** **test_with_several_news_content_by_one_user**(self):
        news1 **=** NewsContent**.**objects**.**create(
            headline**=**'Test Title 1',
            body**=**'Test Body 1',
            author**=**self**.**author
        )
        news2 **=** NewsContent**.**objects**.**create(
            headline**=**'Test Title 2',
            body**=**'Test Body 2',
            author**=**self**.**author
        )
        news3 **=** NewsContent**.**objects**.**create(
            headline**=**'Test Title 3',
            body**=**'Test Body 3',
            author**=**self**.**author
        )
        news4 **=** NewsContent**.**objects**.**create(
            headline**=**'Test Title 4',
            body**=**'Test Body 4',
            author**=**self**.**author
        )
        news5 **=** NewsContent**.**objects**.**create(
            headline**=**'Test Title 5',
            body**=**'Test Body 5',
            author**=**self**.**author
        ) 

        response **=** self**.**client**.**get(self**.**url)

        self**.**assertEqual(200, response**.**status_code)
        self**.**assertContains(response, news1**.**headline)
        self**.**assertContains(response, news2**.**headline)
        self**.**assertContains(response, news3**.**headline)
        self**.**assertContains(response, news4**.**headline)
        self**.**assertContains(response, news5**.**headline)
```

# 让我们开始重构

首先，为什么使用 **faker** 那是一个 **Python** 包，为你生成假数据。我们的测试是如何将**变成**的:

```
*# in tests/test_views.py*
**from** django.contrib.auth **import** get_user_model
**from** django.test **import** TestCase
**from** django.urls **import** reverse
**from** faker **import** Factory

**from** .models **import** NewsContent

User **=** get_user_model()

faker **=** Factory**.**create()

**class** **NewsContentListViewTests**(TestCase):
    **def** **setUp**(self):
        self**.**author **=** User**.**objects**.**create_user(
                          faker**.**name(), 
                          faker**.**email()
                      )
        self**.**news_content **=** NewsContent**.**objects**.**create(
            headline**=**faker**.**word(),
            body**=**faker**.**text(),
            author**=**self**.**author
        )
        self.url **=** reverse(
            'news:news_content_list', 
             args**=**(self**.**author**.**id,)
        )

    **def** **test_with_several_news_content_by_one_user**(self):
        news1 **=** NewsContent**.**objects**.**create(
            headline**=**faker**.**word(),
            body**=**faker**.**text(),
            author**=**self**.**author
        )
        news2 **=** NewsContent**.**objects**.**create(
            headline**=**faker**.**word(), 
            body**=**faker**.**text(), 
            author**=**self**.**author
        )
        news3 **=** NewsContent**.**objects**.**create(
            headline**=**faker**.**word(), 
            body**=**faker**.**text(), 
            author**=**self**.**author
        )
        news4 **=** NewsContent**.**objects**.**create(
            headline**=**faker**.**word(), 
            body**=**faker**.**text(), 
            author**=**self**.**author
        )
        news5 **=** NewsContent**.**objects**.**create(
            headline**=**faker**.**word(), 
            body**=**faker**.**text(), 
            author**=**self**.**author
        )

        response **=** self**.**client**.**get(self**.**url)

        self**.**assertEqual(200, response**.**status_code)
        self**.**assertContains(response, news1**.**headline)
        self**.**assertContains(response, news2**.**headline)
        self**.**assertContains(response, news3**.**headline)
        self**.**assertContains(response, news4**.**headline)
        self**.**assertContains(response, news5**.**headline)
```

想了解更多关于 faker 的有用方法，可以访问 [it](https://faker.readthedocs.io/en/master/providers.html) 。我们有和以前一样的测试功能。Faker 正在随机地**生成值**。现在创建一些工厂。工厂是 Python 类，像 Django 模型一样写入数据库。

```
Model**.**objects**.**create()**or**ModelFactory()
```

让我们为我们的模型创建工厂:

```
**import** factory
**from** django.contrib.auth **import** get_user_model
**from** faker **import** Factory**from** my_news.news.models **import** NewsContentUser **=** get_user_model()faker **=** Factory**.**create() **class** **UserFactory**(factory**.**DjangoModelFactory):
    **class** **Meta**:
        model **=** User name **=** faker**.**name()
    email **=** faker**.**email() **class** **NewsContentFactory**(factory**.**DjangoModelFactory):
    **class** **Meta**:
        model **=** NewsContent headline **=** faker**.**word()
    body **=** faker**.**text()
    author **=** factory**.**SubFactory(UserFactory)
```

*   **类元**:定义你工厂的型号
*   **faker** :生成随机值
*   `factory.SubFactory(NewsContentFactory) == models.ForeignKey(NewsContent)`

```
*# in tests/test_views.py*
**from** django.contrib.auth **import** get_user_model
**from** django.test **import** TestCase
**from** django.urls **import** reverse
**from** faker **import** Factory

**from** .factories **import** NewsContentFactory, UserFactory
**from** .models **import** NewsContent

User **=** get_user_model()

faker **=** Factory**.**create()

**class** **NewsContentListViewTests**(TestCase):
    **def** **setUp**(self):
        self**.**author **=** UserFactory()
        self**.**news_content **=** NewsContentFactory(author**=**self**.**author)
        self.url **=** reverse(
            'news:news_content_list', 
             args**=**(self**.**author**.**id,)
        )

    **def** **test_with_several_news_content_by_one_user**(self):
        news1 **=** NewsContentFactory(author**=**self**.**author)
        news2 **=** NewsContentFactory(author**=**self**.**author)
        news3 **=** NewsContentFactory(author**=**self**.**author)
        news4 **=** NewsContentFactory(author**=**self**.**author)
        news5 **=** NewsContentFactory(author**=**self**.**author)

        response **=** self**.**client**.**get(self**.**url)

        self**.**assertEqual(200, response**.**status_code)
        self**.**assertContains(response, news1**.**headline)
        self**.**assertContains(response, news2**.**headline)
        self**.**assertContains(response, news3**.**headline)
        self**.**assertContains(response, news4**.**headline)
        self**.**assertContains(response, news5**.**headline)
```

# 干(不要重复自己)——工厂

工厂的行为就像来自 ORM 的对象。

*   **创建批处理**:创建一批对象，无需重复我们的代码。

```
*# in tests/test_views.py*
**from** django.contrib.auth **import** get_user_model
**from** django.test **import** TestCase
**from** django.urls **import** reverse
**from** faker **import** Factory

**from** .factories **import** NewsContentFactory, UserFactory
**from** .models **import** NewsContent

User **=** get_user_model()

faker **=** Factory**.**create()

**class** **NewsContentListViewTests**(TestCase):
    **def** **setUp**(self):
        self**.**author **=** UserFactory()
        self**.**news_content **=** NewsContentFactory(author**=**self**.**author)
        self.url **=** reverse(
            'news:news_content_list', 
             args**=**(self**.**author**.**id,)
        )

    **def** **test_with_several_news_content_by_one_user**(self):
        news_contents **=** NewsContentFactory**.**create_batch(
                             5, 
                             author**=**self**.**author
                        )

        response **=** self**.**client**.**get(self**.**url)
        self**.**assertEqual(200, response**.**status_code)
        **for** news_content **in** news_contents:
            self**.**assertContains(response, news_content**.**headline)
```

# 结论

在本文中，我们学习了如何使用 faker 和工厂。它们对模型测试很有帮助。如果你想看更多关于` **factory_boy** 的信息，可以访问[这个](https://factoryboy.readthedocs.io/en/latest/index.html)。