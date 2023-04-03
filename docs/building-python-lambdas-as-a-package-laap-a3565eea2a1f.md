# 将 Python Lambdas 构建为一个包(LaaP)

> 原文：<https://medium.com/analytics-vidhya/building-python-lambdas-as-a-package-laap-a3565eea2a1f?source=collection_archive---------18----------------------->

![](img/c95c3f3a989052aef24d83f3986dfa91.png)

有很多关于构建 Python Lambdas 的文章。有些甚至在 [AWS 文档](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python-how-to-create-deployment-package.html)中。令人恼火的是，这些例子中有许多没有考虑到在生产代码中最好将代码与测试放在一起，以及许多不必要的复制步骤来移动已安装的需求。

使用 pip，再加上它的命令行参数，你可以免费得到很多东西。那么，我们如何开始，创建一个演示目录:

```
mkdir demo_lambda
cd demo_lambda
```

添加一个模块，这将包含我们的 Lambda 函数:

```
mkdir todo_lambda
touch todo_lambda/__init__.py
```

现在，让我们编写我们的 Lambda 函数，把它放在`todo_lambda/todo_lambda_function.py`中:

相当简单！它将调用 [JSONPlaceholder](https://jsonplaceholder.typicode.com/) 并获得一个 todo。为了使 Lambda 的路径更简单，在`todo_lambda/__init__.py`中添加一个相对导入:

这意味着我们现在可以把处理者称为`todo_lambda.todo_handler`。这在设置 Lambda 函数时很有用。

现在，作为优秀的编码人员，让我们编写一些测试，首先制作测试模块:

```
mkdir tests
touch tests/__init__.py
```

让我们在`tests/test_todo_lambda_function.py`中编写一个简单的测试:

你现在可以用`pytest`从你的根目录运行这些。

所以你可能会说“酷，一些简单的 Python 代码，那又怎么样？”，嗯，我们还没有到酷的地步，以上都是标准。这才是重点。

## 好的一面

对，我们到了，在好地方。让我们把这个代码打包。添加您的`setup.py`:

现在来制作拉链。首先，进行安装:

```
python3 -m pip install . -t output/install
```

这将把所有的 Lambdas 需求安装到安装目录中。那是`-t output/install`位。现在让我们闭嘴:

```
cd output/install
zip -r ../lambda.zip *
cd -
```

我们现在有一个 Lambda zip，现在让我们发布它。首先，创建角色:

```
aws iam create-role --role-name lambda-role --assume-role-policy-document '{"Version": "2012-10-17","Statement": [{ "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]}'
```

并附加一个基本的 lambda 策略:

```
aws iam attach-role-policy --role-name lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

> 我们在这里偷懒，使用管理策略。显然，您会创建自己的策略，并可能使用一些**基础设施作为代码**工具，如`*terraform*`。

使用以下内容发布 Lambda 函数:

```
aws lambda create-function --function-name todo-lambda --runtime python3.7 --role arn:aws:iam::${AWS_ACCOUNT_ID}:role/lambda-role --handler todo_lambda.todo_handler --zip-file fileb://output/lambda.zip --region eu-west-1
```

> `*export AWS_ACCOUNT_ID=$(aws sts get-caller-identity | jq -r .Account)*`在这里知道这是一个很好的命令。

我们现在可以用以下方式调用它:

```
aws lambda invoke --function-name todo-lambda --payload '{"todo_id": 1}' response.json
```

如果一切顺利，您的`response.json`应该是这样的:

```
{
  "userId": 1,
  "id": 1,
  "title": "delectus aut autem",
  "completed": false
}
```

## 结尾注释

使用这种方法，您可以以一种简洁的方式编写代码，进行良好的测试，并使其易于部署。这是一种模式，您可以很容易地在您的代码库中复制，并将其用作一个好的标准。

从这里开始，你可以玩其他东西:

*   Pip 安装也可以使用需求文件，比如`pip install . -r requirements.txt`，那些需求文件可以有需求文件…有戏！
*   试着把这个土改一下，从长远来看会有很大帮助。
*   Pip 是 Python，这意味着你可以从 Python 运行它…请注意 pip 变化很大，你可能需要在更新前测试版本。
*   集成测试总是可以放在一个单独的目录中，它们也可以放在一起…很好！