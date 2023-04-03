# C++智能指针和引用

> 原文：<https://medium.com/analytics-vidhya/c-smart-pointers-and-references-4b15ee5d14bf?source=collection_archive---------11----------------------->

# 共享指针

*共享*指针是一种智能指针，允许数据的分布式所有权。每分配一次，一个*引用计数*就增加一个，表示数据多了一个所有者。当指针超出范围或所有者调用`reset`时，引用计数递减。当引用计数变为 0 时，指向的数据被释放。使用`make_shared`:

```
auto person = std::make_shared<Person>();
if(person)
{
    cout << "Person's address is " << person->address;
}
```

C++17 允许使用共享指针指向动态分配的数组。但是，`make_shared`不能用。下面是一个例子:

```
shared_ptr<Person[]> persons(new Person[10]);
persons[0]._name = "Jack Sparrow";
persons[0]._address = "Caribbean";
cout << "Address of first person "<<persons[0].address << endl;
```

一个更好的解决方案是使用`std::vector`，因为它可以动态地增长和收缩。

```
std::vector<std::shared_ptr<Person>> persons;
auto person1 = std::make_shared<Person>("Jack Sparrow","Caribbean");
auto person2 = std::make_shared<Person>("Hector Barbossa","Bahamas");
persons.push_back(person1);
persons.push_back(person2);for (auto& person : persons)
{
    std::cout << "Person name is " << person->_name << " and address is " << person->_address << "\n";
}
```

# 常数

`const`的第一个用途是声明和定义一旦初始化就不能改变的值。这些宏取代了 c 语言中的`#define`宏常量。例如:

```
const double PI = 3.14;
const int NUMVALUES = 42;
const std::string quote = "To be or not to be";
```

## 参数常量

第二个可以使用的地方是确保参数不能被修改。例如，如果一个函数带有一个`int*`指针，那么用`const`传递指针可以确保函数不会改变值。示例:

```
int main()
{
    int num = 5;
    int *val = &num;
    foo(val);
}void foo(const int* value)
{
    *value = 6; //will fail to  compile
}
```

# 参考

引用是现有变量的别名。示例:

```
double aDouble = 3.14
double& refToDouble = aDouble;
```

引用的特殊之处在于，人们仍然像使用普通变量一样使用它们，但在幕后，它们是指向原始变量的“指针”。因此，修改引用与修改原始变量是一样的。上接示例:

```
refToDouble = 6.0;
cout << "Double value is " << aDouble << endl; //prints out 6.0
```

## 通过引用传递

当参数被传递给函数时，变量的一个*副本*被制作，该副本在函数内被修改。为了修改原始变量，C 习语将一个*指针*传递给变量。示例:

```
int main()
{
    int x = 5;
    int y = 3;
    swap(&x, &y);
    printf("x is %d and y is %d",x,y); //prints out 3 and 5
}
void swap(int *a , int *b)
{
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
```

C++通过使用*引用传递*来避免使用指针。在 C++中，相同的交换函数实现如下:

```
int main()
{
    int a = 2;
    int b = 3;
    swap(a,b);
}
void swap(int &a , int &b)
{
    int temp;
    temp = a;
    a = b;
    b = temp;
}
```

注意，语法允许引用作为普通变量使用，但是原始变量被修改了。

当返回一个大的结构或类对象时，传统的习语是传入一个对结构或对象的非常数引用，并让函数修改它。C++11 移动语义消除了这种需要。

通过`const`参考

当你需要有效地传递一个值给一个函数，但确保它不被修改时，使用 pass by `const`引用。示例:

```
void foo(const int& a)
{ 
    int b = a * 2;
    a++; //compiler will throw an error here as one attempts to modify a.
}
```

当通过`const`引用传递`std::string`时，传入字符串文字有效。

# 类型推理

类型推断是编译器推断表达式类型的机制。两个关键字，`auto`和`decltype`在 C++中用于类型推断。`auto`用法如下:

```
auto x = 2 + 3;
```

当处理函数返回的复杂类型时，使用`auto`的优势变得很明显。注意，使用`auto`去掉了引用和常量引用限定符并进行复制，所以如果一个函数返回一个引用，一定要使用`auto&`。类似地，使用`const auto&`作为`const`参考。

示例:

```
int& bar()
{ 
    int* dynInt = new int;
    *dynInt = 5;
    return *dynInt;
}auto retVal = bar(); //A copy is made here. Use auto& instead.
```

`decltype`将表达式作为参数，并推断该表达式的类型。优点是引用和常量引用不会被剥离。示例:

```
decltype(bar()) retVal = bar();
```

# 班级

一个类模拟真实世界和抽象对象，可以被看作是制作对象的蓝图。类是在头文件中声明的。h)文件及其定义位于. cpp 文件中。下面是一个类的示例:

`checkingaccount.h`:

```
#ifndef BANKACCOUNT_H
#define BANKACCOUNT_H
#include <string>
using std::string;
class CheckingAccount 
{ 
    public:
        CheckingAccount(string accountHolder, string accountNumber, double balance);
        double getBalance() const;
        string getAccountHolder() const;
        string getAccountNumber() const;
        void deposit(double amount);
        string withdraw(double amount);
        ~CheckingAccount(); private:
        string mAccountHolder;
        string mAccountNumber;
        double mBalance;
}
#endif
```

`checkingaccount.cpp`:

```
#include "checkingaccount.h"CheckingAccount::CheckingAccount(string accountHolder, string accountNumber, double balance)
{ 
    mAccountHolder = accountHolder;
    mAccountNumber = accountNumber;
    mBalance = balance;
}double CheckingAccount::getBalance() const
{
    return mBalance;
}string CheckingAccount::getAccountHolder() const
{
    return mAccountHolder;
}string CheckingAccount::getAccountNumber() const
{
    return mAccountNumber;
}void CheckingAccount::deposit(double amount)
{
    mBalance += amount;
}string CheckingAccount::withdraw(double amount)
{
    if(mBalance - amount <= 0.0)
    {
        return "Insufficient Funds";
    }
    mBalance -= amount;
    return "Amount withdrawn successfully";
}CheckingAccount::~CheckingAccount()
{
    //no cleanup for this case
}
```

**构造函数**是一个与没有返回类型的类同名的方法。当在堆栈或堆上初始化一个对象时调用它。当堆栈上的对象超出范围时，或者当在堆分配的对象上调用`delete`时，调用**析构函数**。构造函数负责初始化类，而析构函数负责清理，包括关闭文件句柄、释放对象分配的内存等等。这是标有`~`的功能。

**构造函数初始化器**是首选的初始化方法。其语法如下:

```
CheckingAccount::CheckingAccount(string accountHolder, long accountNumber, double balance)
    :mAccountHolder(accountHolder),
     mAccountNumber(accountNumber),
     mBalance(balance)
{ 
}
```

## 使用类

使用类的示例如下:

```
#include "checkingaccount.h"
#include <memory>
using std::unique_ptr;
using std::cout;
using std::endl;int main()
{
    CheckingAccount chkAcnt("Jack Sparrow","61723",10000); //allocated on the stack.
    chkAcnt.deposit(100.0);
    chkAcnt.withdraw(50.0);
    auto chkAccntTwo = make_unique<CheckingAccount>("Hector Barbossa",61724,10000);
    chkAccntTwo->deposit(1000.00);
    chkAccntTwo->withdraw(5000.00);
    cout << chkAccountTwo.getAccountHolder() <<"'s balance is "<< chkAccntTwo->getBalance() << endl;
} //chkAcnt destructor is called here. chkAccntTwo destructor also called here as unique_ptr goes out of scope.
```

# 参考资料:

m .格雷瓜尔(2018)。*专业 C++* 。印第安纳州:约翰·威利&之子。

[](https://codingadventures1.blogspot.com/2020/02/c-tour-part-iii.html) [## C++教程第三部分

### 共享指针是一种智能指针，允许数据的分布式所有权。每次它被赋值时，一个…

codingadventures1.blogspot.com](https://codingadventures1.blogspot.com/2020/02/c-tour-part-iii.html)