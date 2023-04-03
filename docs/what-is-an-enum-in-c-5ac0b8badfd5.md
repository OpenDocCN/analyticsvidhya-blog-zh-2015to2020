# 什么是 C 语言中的枚举？

> 原文：<https://medium.com/analytics-vidhya/what-is-an-enum-in-c-5ac0b8badfd5?source=collection_archive---------25----------------------->

C 中的一个 enum 是 ***用来枚举事物*** 。例如，枚举一周中的日期，如星期一、星期二或星期三，或者枚举错误代码，如 OK 或 WARN，或者枚举颜色，如红色或蓝色。列举的数据属于`int`类型。

*枚举关键字*用于声明枚举类型的[标识符](https://twiserandom.com/c/what-is-an-identifier-in-c/)，

```
enum color ; 
enum error_codes;
```

`enum color`和`enum error_codes`是 ***不完整类型*** 因为没有枚举数据被初始化，它们不能在以后被用来声明`enum color`或`enum error_codes`类型的变量。

```
enum color ;
enum color red ;
/* Will cause the following compilation error :
 error: variable has incomplete type 'enum color'
  enum color red ;*/
```

所以一个枚举类型，比如一个`enum color`被用来枚举事物，一个枚举类型通过使用`enum`关键字来声明。一个枚举类型 ***在声明时必须使用枚举器列表列出其枚举数据*** ，否则将是一个不完整的类型。可以这样做:

```
enum daysOfWeek { Monday , Tuesday , Wednesday , Thursday , Friday , Saturday , Sunday };
```

前面的声明将标识符`daysOfWeek`声明为`enum`类型，它用`int`类型的枚举数据初始化:星期一、星期二、星期三、星期四、星期五、星期六和星期天。星期一将有一个值`0`，星期二有一个值`1`，并且 ***每个后续数据将有一个值*** 为前一个加一。

```
#include<stdio.h>enum daysOfWeek  { Monday , Tuesday , Wednesday , Thursday , Friday , Saturday , Sunday };int main ( void ) {printf( "Monday is : %d \n", Monday );
  printf( "Tuesday is : %d \n", Tuesday );
  printf( "Wednesday is : %d \n", Wednesday );
  printf( "Thursday is : %d \n", Thursday );
  printf( "Friday is : %d \n", Friday );
  printf( "Saturday is : %d \n", Saturday );
  printf( "Sunday is : %d \n", Sunday );}/* output :
Monday is : 0 
Tuesday is : 1 
Wednesday is : 2 
Thursday is : 3 
Friday is : 4 
Saturday is : 5 
Sunday is : 6 */
```

枚举数据，比如星期一、星期二或星期五…有一个范围，在这个范围内这些标识符是可见的。它们的作用域是被声明为 的 ***作用域，所以如果它们被声明为全局作用域，就像在前面的例子中一样，它们有一个源文件或全局作用域，因此它们可以从源文件中的任何地方被访问，如果在函数中声明，它们有一个[块作用域](https://twiserandom.com/c/what-is-an-identifier-in-c/#Scope_of_an_identifier)，所以它们只能从函数内部被访问…***

枚举数据，也被称为 ***枚举常数*** ，因为一旦它在枚举器列表中被赋值，它的值就不能改变。

***不需要为 enum 类型*** 提供标识符，只需要初始化数据即可。

```
#include<stdio.h>int main ( void ) {
  enum  { Monday , Tuesday , Wednesday , Thursday , Friday , Saturday , Sunday };
  printf( "Tuesday is : %d \n" , Tuesday );}/* output :
Tuesday is : 1*/
```

初始化数据，可以使用`=`操作符初始化为 ***特定值*** 。下一个枚举的数据将总是具有前一个数据加 1 的值。除此之外，枚举数据可以被初始化为具有相同的值。

```
#include<stdio.h>enum code { OK , WARN, ERROR , PASS = 0 , INFORMATIONAL , EXCEPTION };int main ( void ) {
    printf( "[OK,%d] , [WARN,%d] , [ERROR,%d]\n" , OK , WARN , ERROR  );
    printf( "[PASS,%d] , [INFORMATIONAL,%d] , [EXCEPTION,%d]\n" , PASS , INFORMATIONAL , EXCEPTION );}/* output :
[OK,0] , [WARN,1] , [ERROR,2]
[PASS,0] , [INFORMATIONAL,1] , [EXCEPTION,2]
*/
```

枚举类型 的 ***变量可以使用以下方法之一进行声明:***

```
enum color{red , green , blue}aFirstColor ;
enum color aSecondColor ; 
enum {OK, WARN }status_code ;
```

在 ***的第一个方法*** 中，枚举类型`color`的变量 `aFirstColor`在声明和初始化`enum color`类型时被声明。

在 ***第二个方法*** 中，变量`aSecondColor`是在声明了枚举类型之后声明的。并且在 ***最后一个方法*** 中，声明了一个变量`status_code`而没有给 enum 类型分配标识符。

***枚举类型*** 的变量不必将枚举值之一作为值，它可以被赋予任何值。

```
#include<stdio.h>enum networkState { up , down };int main ( void ) {
  enum networkState state = 2 ;
  enum networkState *statePtr = &state;
  switch( *statePtr )
    {
    case up :
      printf( "network state is up\n" );
      break;
    case down :
      printf( "network state is down\n" ) ;
      break ;
    default :
      printf( "network state is unknown\n" );
  }}/* output :
network state is unknown */
```

*原载于 2020 年 12 月 13 日 https://twiserandom.com**[*。*](https://twiserandom.com/c/what-is-an-enum-in-c/)*