# C++类和对象

> 原文：<https://medium.com/analytics-vidhya/c-classes-and-objects-5dae9b3d9fd3?source=collection_archive---------21----------------------->

![](img/4bf0fa48f423a21323668ca966f2b36a.png)

本文将使用一个`CheckingAccount`类作为运行示例。

# 写作课

编写类时，必须考虑由类表示的*行为*和*数据成员*(数据)。一个类就像一张蓝图，而对象就像根据蓝图建造的建筑。

## 类别定义

下面是`CheckingAccount`类的第一个剪辑:

`checkingaccount.h`:

```
#ifndef CHECKING_H
#define CHECKING_H
class CheckingAccount
{
    public:
        CheckingAccount(double balance);
        ~CheckingAccount();
        void deposit(double amount);
        bool withdraw(double amount);
        double getBalance() const;
    private:
        double mBalance;
} 
#endif
```

`checkingaccount.cpp`:注意`::`操作符是*范围解析操作符*。

```
CheckingAccount::CheckingAccount(double balance) //constructor
{
    mBalance = balance;
}void CheckingAccount::deposit(double amount)
{
    mBalance += amount;
}bool CheckingAccount::withdraw(double amount)
{
    if((mBalance - amount) > 0)
    {
        mBalance -= amount;
        return true;
    }
    return false;
}double getBalance() const
{
    return mBalance;
}
//Destructor not implemented as there is no dynamic memory allocated.
```

## 访问控制

*   `public`:任何代码都可以调用`public`成员函数或者访问`public`数据成员。
*   `protected`:只有类及其派生类才能访问成员函数或数据成员
*   `private`:只有类可以访问成员函数或数据成员。

## 类内成员初始值设定项

可以像这样直接在类定义中初始化成员变量:

```
class CheckingAccount
//....
private : 
    double mBalance = 0.0;
```

## `this`指针

类的每个成员函数都有一个名为`this`的隐式指针参数，它指向当前对象。它可用于从数据成员中消除参数的歧义。它还可以用于将对当前对象的引用传递给一个函数，该函数接受对该对象的引用或常量引用。

## 堆栈上的对象

C++允许在堆栈和堆上分配对象。示例:

```
CheckingAccount chkAccount(1000.00);
cout << "Balance is " << chkAccount.getBalance() << endl;
```

## 堆/空闲存储区上的对象

可以使用原始指针或智能指针之一在堆上分配对象。示例:

```
auto myChkAccount = std::make_unique<CheckingAccount>(10000.00); //smart pointer
cout << myChkAccount->getBalance() << endl;CheckingAccount * chkaccount = new CheckingAccount(10000.00); //prefer smart pointer over this
cout << chkaccount->getBalance() << endl;
delete chkaccount;
```

# 对象生命周期

对象生命周期由*创建*、*销毁*和*分配*组成。

## 创造

当一个对象被创建时，它嵌入的所有对象也被创建。示例:

```
#include <string>class Foo
{
    private: 
        std::string mAddress;
}int main()
{
    Foo fooObj;
    return 0;
}
```

`string`对象在`fooObj`创建时创建，在`fooObj`析构时析构(在这种情况下，超出范围)。

## 构造器

构造函数是一种特殊的成员函数，用于初始化类的值。默认构造函数不接受任何参数，或者所有数据成员都被赋予默认值。

## 堆栈上的构造函数

```
CheckingAccount account(5000.00);
```

## 堆上的构造函数

```
auto chkAccount = std::make_unique<CheckingAccount>(5000.00);
CheckingAccount anotherAccount = nullptr;
anotherAccount = new CheckingAccount(100.00);
```

## 当需要默认构造函数时

创建对象数组时，需要默认的构造函数，因为没有调用任何其他构造函数的选项:

```
CheckingAccount accounts[5];
CheckingAccount* accounts = new CheckingAccount[5];
```

默认构造函数的示例如下:

```
CheckingAccount::CheckingAccount()
{
    mBalance = 0.0;
}
```

请注意，在堆栈上使用默认构造函数时，必须省略括号，如下所示:

```
CheckingAccount chkAccount;
chkAccount.deposit(100.00);
```

如果程序员没有编写默认的构造函数，编译器会生成一个。但是，如果编程了任何构造函数，则编译器会省略构造函数的生成。

## 显式删除的构造函数

如果您只有静态方法，并且不想要构造函数或由编译器生成构造函数，请执行以下操作:

```
class CheckingAccount
{
    public: 
       CheckingAccount() = delete;
}
```

## 构造函数初始值设定项

有一种替代方法来初始化构造函数中的数据成员，称为*构造函数初始化器*:

```
CheckingAccount:: CheckingAccount(double balance) : mBalance(balance)
{ 
}
```

当 C++创建一个对象时，必须调用嵌入对象的构造函数本身。构造函数初始化器(ctor-initializer)允许调用这些构造函数，因为在构造函数体中，值被修改，但没有初始化。

如果一个嵌入对象有一个默认的构造函数，就不需要在*构造函数初始化器*中初始化它。否则，初始化*初始化器*中的对象。有些类型*必须在*构造器初始化器*中初始化*，如下所示:

*   `const`数据成员:只能创建和分配一次。
*   引用:只能在引用变量时存在
*   没有默认构造函数的嵌入对象
*   没有默认构造函数的基类

请注意，构造函数初始化器按照数据成员在类定义中出现的顺序初始化数据成员，而不是按照它们在初始化器列表中出现的顺序。

# 复制构造函数

*复制构造函数*让你创建一个对象，它是另一个对象的精确副本。如果程序员没有提供，编译器会生成一个，从源对象中相应的数据成员初始化每个数据成员。
对于嵌入对象，调用它们的复制构造函数。示例:

```
class CheckingAccount
{
    public:
       CheckingAccount(const CheckingAccount &src);
}
```

实施:

```
CheckingAccount::CheckingAccount(const CheckingAccount &src) : mBalance(src.mBalance)
{}
```

给定数据成员`n1`、`n2`、`nM`，编译器生成一个复制构造函数，如下所示:

```
cname::cname(const cname& src)
: n1(src.n1), n2(src.n2),...,nM(src.nM) {}
```

因此，在许多情况下，没有必要显式指定复制构造函数。

## 当调用 Copy Ctor 时

每当对象通过值传递给函数或方法时，就调用复制构造函数。示例:

```
void foo(std::string name)
{
    cout << "Name is "<<name<<endl;
}int main()
{
    string name = "John";
    foo(name); //copy constructor
}
```

## 当显式调用复制构造函数时

示例:

```
CheckingAccount account(5000.00);
CheckingAccount accountCopy(account);
```

## 通过引用传递对象

以下是指导方针:

*   通过常量引用传递对象以提高性能(除非需要修改)
*   按值传递原语
*   通过值传递`string_view`,因为它只是一个指针和长度
*   不要传递对堆栈上对象的引用。还一份。

要禁止通过值传递对象，请删除复制构造函数:

```
CheckingAccount(const CheckingAccount &src) = delete;
```

## 初始值设定项列表构造函数

*initializerlist 构造函数*是一个第一个参数为`std::initializer_list<T>`的构造函数，没有附加参数或者附加参数给定默认值。

```
class Sequence
{
    public :
        Sequence(initializer_list<int> params)
        { 
            for(const auto& value: params)
            {
                values.push_back(value); //note push_back takes const reference parameter and makes a copy of it internally
            }
        }
    private:
        vector<int> values;
}
```

## 委托构造函数

委托构造函数使构造函数能够从 ctor-initializer(它必须是唯一的成员初始值设定项)内部调用其他构造函数。

# 对象销毁

当一个对象被销毁时，对象的析构函数方法被调用，如果析构函数被正确实现，分配的内存被释放。如果程序员不创建析构函数，
编译器会创建一个进行递归成员式析构的析构函数。堆栈上的回调对象在超出范围时被销毁。堆栈上的对象按构造的相反顺序销毁。

# 重载赋值运算符

重载赋值运算符不同于实现复制构造函数，因为“复制”只在对象初始化时发生。该运算符也称为复制赋值运算符。
举例:

```
CheckingAccount& operator=(const CheckingAccount &rhs);
```

返回对该对象的引用以允许*链接赋值*。重载赋值运算符的示例:

```
CheckingAccount& CheckingAccount::operator=(const CheckingAccount& rhs)
{
    if(this == &rhs)
    {
        return *this; //have to do this as self assignment is allowed in C++
    }
    mBalance = rhs.mBalance;
    return *this;
}
```

# 区分拷贝和转让

复制结构的示例:

```
CheckingAccount acct1(10.0);
CheckingAccount acct2(acct1);CheckingAccount acct3 = acct2;
```

分配示例:

```
acct3 = acct1;
```

# 对象作为返回值

例如，如果从函数中返回一个`std::string`并按如下方式赋值，则调用复制构造函数，然后调用赋值操作符:

```
string s1;
s1 = getString();
```

在这种情况下，调用了两个复制构造函数:

```
string s1 = getString(); //getString calls copy constructor and s1's copy constructor is called.
```

编译器经常做*返回值优化(RVO)* 来消除对返回值的复制构造函数调用。

# 复制构造函数和对象数据成员

编译器生成的复制构造函数递归调用每个嵌入对象的复制构造函数。然而，当程序员实现时，只有使用 ctor-initializer 才能确保调用嵌入对象的复制构造函数:

```
CheckingAccount::CheckingAccount(const CheckingAccount &src) : mBalance(src.mBalance) {}
```

如果改为执行以下操作，则使用赋值运算符，因为编译器会在主体执行时调用默认的构造函数:

```
CheckingAccount::CheckingAccount(const CheckingAccount &src)
{
    mBalance = src.mBalance; //assignment operator
}
```

# 参考资料:

m .格雷瓜尔(2018)。*专业 C++* 。印第安纳州，约翰·威利的儿子们。

[](https://codingadventures1.blogspot.com/2020/02/classes-and-objects.html) [## 类别和对象

### 本文将使用 CheckingAccount 类作为运行示例。编写类时，必须考虑…

codingadventures1.blogspot.com](https://codingadventures1.blogspot.com/2020/02/classes-and-objects.html)