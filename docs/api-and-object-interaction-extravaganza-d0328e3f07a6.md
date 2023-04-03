# Ruby 中的 API 和对象交互盛会

> 原文：<https://medium.com/analytics-vidhya/api-and-object-interaction-extravaganza-d0328e3f07a6?source=collection_archive---------24----------------------->

## 第二部分:从执行到整合

## 上接[第 1 部分:从沙箱到 API 设置](/@chaserabenn/api-and-object-interaction-extravaganza-a0ce928fbc2a)

第二部分各节

*   [Ruby API 第五步:使用 API](#f4e8)
*   [Ruby API 第六步:让它动态化](#e1da)
*   [结论](#372b)

## 第五步:

是时候用我们的端点创造一些奇迹了。

![](img/d4aa290f027dc0e240d86d0035161da1.png)

尼古拉斯·天梭在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

在此之前，我们应该先弄清楚一些定义(一些新的，一些回顾)。为了理解这篇文章的其余部分，下面的定义过于简单了。这些术语中的每一个都可以成为他们自己博客文章的主题，所以不要觉得你现在需要完全理解它们(因为我不需要)。你只需要相信，在这种情况下，这些事情中的每一件都做了重要的事情。

```
GET Request: an online request that our system sends to another systems that hosts information we are looking for.Parse: when you break something down into parts so they can be read and manipulated. E.g. Breaking a sentence down into words and those words into individual characters.Nested Data Structure: a sometimes maddening mix of hashes and arrays inside of each other. The JSON-parsed (see below) response to a GET request will be some form of nested data. Think of it as the coding equivalent of a Russian Nesting Doll.Web Resource: Anything that can be obtained from the web and uniquely identified. E.g. that one puppy photo you love.URI: Uniform Resource Identifier. A string of characters that identifies and names a web resource. Also, a Ruby Module with a library of methods and functions. When you see lowercase "uri", it refers to the web resource identifier and when you see uppercase "URI", it refers to the Ruby Module.URL: Uniform Resource Locater. A web address that is the location of a web resource on a network. Since it is a unique location, the URL for a resource is also technically its name.Net::HTTP : A built-in Ruby class that has a library of functions and methods that can be used to build user-agents like URI.Net::HTTPOK : Ruby Class and the type of object that is returned when a GET request is called with Net::HTTP. It has a method “body” that you will be using. Loves enthusiastically saying "OK" after regular Net::HTTP has done all the work.JSON: JavaScript Object Notation. A complex, yet human-readable, way of storing, transferring, and presenting the nested data from the GET response. Think of it as the drill sergeant of the GET Request response. JSON marches in and gets everyone in order.
```

请记住，所有这些的目标是获得您的程序可以分解和操作的数据。我们可以尝试构建一个自定义的翻译器方法来对 GET 响应进行排序，或者我们可以依赖 Ruby 已经为我们制作的内置方法库。我想我会选择后者。

请注意，列出的 apiKey 不是真实的，因此您将无法向 Spoontacular 创建真实的 GET 请求，除非您注册了一个帐户。

设置您的系统以使用 URI 和 Net::HTTP

```
require ‘open-uri’require ‘net/http’
```

将端点 url 保存到变量“url”

```
url = “https://api.spoonacular.com/recipes/findByIngredients?apiKey=123abc123abc&ingredients=onion+pepper&number=10” 
```

调用 URI 模块上的 parse 方法，并传入变量“url”。结果保存到变量“uri”换句话说，我们解析 url 并得到一个 uri。

```
uri = URI.parse(url)# => #<URI::HTTPS https://api.spoonacular.com/recipes/findByIngredients?apiKey=123abc123abc&ingredients=onion+cheese&number=10&ranking=2>
```

用 Net::HTTP.get_response 创建一个 GET 请求，传入变量“uri”。这个 GET 请求的返回值(或响应)是一个 Net::HTTPOK 对象，我们将把它保存到变量“response”中

```
response = Net::HTTP.get_response(uri)#=> #<Net::HTTPOK 200 OK readbody=true>
```

我们在“response”上调用 body 方法，它从响应中检索正文文本。这样做是因为响应的主体包含了我们想要的数据。

```
response_body = response.body#=>”[{\”id\”:514079,\”title\”:\”Easy Pinwheel Steaks with Spinach and Cream Cheese\”,\”image\”:\”https://spoonacular.com/recipeImages/514079-312x231.jpg\",\"imageType\":\"jpg\",\"usedIngredientCount\":1,\"missedIngredientCount\":2,\"missedIngredients\ . . . ”
```

哇，真是一团糟。如您所见，此时，response_body 是一个混合了各种格式符号的嵌套数据字符串。这不太好。这些杂乱的字符包含了我们想要的数据，但是以一种不可访问的方式。幸运的是，我们有一个清理响应的方法。

JSON 来拯救世界了。

```
require ’json’json_response = JSON.parse(response_body)#=> [
     {“id”=>514079, 
     “title”=>”Easy Pinwheel Steaks with Spinach and Cream Cheese”, 
     “image”=>”https://spoonacular.com/recipeImages/514079-312x231.jpg", 
     “imageType”=>”jpg”, 
     “usedIngredientCount”=>1, 
     “missedIngredientCount”=>2, 
     “missedIngredients”=>
         [{“id”=>11457, 
           “amount”=>4.0,
           “unit”=>”oz”, 
           “unitLong”=>”ounces”, 
           “unitShort”=>”oz”, 
           “aisle”=>”Produce”, 
           “name”=>”baby spinach leaves”, 
            “original”=>”4 oz. fresh baby spinach leaves”,        
            “originalString”=>”4 oz. fresh baby spinach leaves”}]}]
```

漂亮！终于有我们可以合作的东西了。我们使用了一个为处理 JSON 而设计的 Ruby 库，并调用该库的“parse”方法来获取 response_body，并去掉所有妨碍我们访问和理解所请求数据的格式化符号。

## 第六步:

是时候把这一切都集中起来了。

上面的代码很棒，但它是硬编码的。如果您计划在每个 GET 请求中手动输入用户选择的配料，那么您最好不要编写程序，而让用户直接向您发送食谱信息。这个想法是将自动化引入这个过程

为了演示如何将上面的代码构建到您的应用程序中，我们将浏览 UIOLI 的代码片段

我们需要一个在被调用时专门发出 GET 请求的类。还记得沙盒里的 OlderKid 吗？这就是 OlderKid 变成的样子。然而，将这个调用称为 OlderKid 是没有意义的，所以我们将使用 GetRequester。GetRequester 将接收一个 url，并在我们刚刚讨论的所有代码中运行它。

```
require ‘net/http’require ‘open-uri’require ‘json’ class GetRequester def initialize(url) @url = url end def get_response_body uri = URI.parse(@url) response = Net::HTTP.get_response(uri) response.body
    end def parse_json JSON.parse(self.get_response_body) endend
```

有了这样定义的类，我们只需要在 GetRequester 的实例上调用#parse_json 方法。这是因为 parse_json 还调用了 GetRequest 类定义中的另一个方法:get_response_body。在一种方法中，我们可以用 uri 解析 url，用 Net::HTTP 创建 GET 请求，在对该请求的响应中调用 body 方法，最后让 JSON 将数据解析为纯 NDS delight。

```
recipe_url = “https://api.spoonacular.com/recipes/findByIngredients?apiKey=123abc123abc&ingredients=onion+pepper&number=10”recipes = GetRequester.new(recipe_url)
```

我们设置了密码。现在我们需要将这个类与用户输入函数连接起来，要么使用它，要么失去它；

我们的应用程序中的 TTY 提示(谢谢 [Grant Yoshitsu](https://medium.com/u/9eba2377ff56?source=post_page-----d0328e3f07a6--------------------------------) )刚刚从用户那里接收了一组配料，并将其保存到一个数组中。

```
uioli_array = ["onion", "pepper"]
```

![](img/9ada89b95ffead2d0977feb0239d8157.png)

[达沃·尼塞维奇](https://unsplash.com/@davornisevic?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

我们需要将从用户那里获得的数据转换成插值所需的形式。根据 API 文件，该表格是由“+”分隔的一串食品项目。我们要寻找的最终结果是“onion+pepper ”,所以我们将对数组使用 join 方法，并将其保存到变量“uioli_items”

```
uioli_items = uioli_array.join(“+”)
```

现在，我们可以使用 recipe_url，它会将格式正确的用户输入粘贴到字符串插值字段原来所在的位置。

```
recipe_url = “https://api.spoonacular.com/recipes/findByIngredients?apiKey=12abc12&ingredients=#{uioli_items}&number=10&ranking=2"
```

太好了！我们的 url 准备好被传递到 GetRequester 的实例中。我们将在实例上调用 parse_json，并将其保存到变量“response”

```
recipe_url = “https://api.spoonacular.com/recipes/findByIngredients?apiKey=12abc12&ingredients=onion+pepper&number=10&ranking=2"response = GetRequester.new(recipe_url).parse_json
```

您可能已经注意到，当我们查看 JSON 解析时，从 GET 返回的 10 个食谱中的每一个都有大量的信息。然而，我们只需要来自每个菜谱的两条信息:名称和 id。出于这个原因，我们将编写一个名为“clean_recipes”的方法来清理嵌套数据，只将我们需要的信息保存到一个散列数组中。

```
recipes = [{“name” => “Chicken Soup”, “website_id” => “19283”}, {“name” => “Spaghetti and Meatballs”, “website_id” => “39290”}]
```

该方法将创建一个空数组，遍历每个配方，将名称和 Id 作为键/值对保存到哈希中，将哈希放入我们创建的数组中，并返回该数组。

```
def clean_recipes(recipes)     recipes =[]     
    url.each do |recipe|         recipe_hash = {}         recipe_hash[“name”] = recipe[“title”].titleize         recipe_hash[“website_id”] = recipe[“id”]         recipes << recipe_hash end recipes end
```

最后，我们将调用 clean_recipes 方法并传入 recipes(对 GET 请求的 json 解析响应)。这将被保存到变量“结果”中。

```
results = clean_recipes(recipes)
```

TTY-Prompt 从这里接管并向用户显示配方名称以供选择。

## 我们已经到达终点。。。

呦喂。这是很多信息。我写这篇文章的目的是提供一些关于 API 的观点和见解。之前，我说过 API 令人兴奋，我是认真的。作为一名程序员新手，学习如何使用 API 让我的应用程序变得生动起来。我的应用程序不再注定要被关进 MacBook 监狱。他们可以与整个宇宙的系统和他们持有的数据进行对话和交互。我们能创造的可能性是无限的。