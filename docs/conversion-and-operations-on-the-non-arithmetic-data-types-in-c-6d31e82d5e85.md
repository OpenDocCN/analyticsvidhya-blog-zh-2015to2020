# C 语言中非算术数据类型的转换和运算

> 原文：<https://medium.com/analytics-vidhya/conversion-and-operations-on-the-non-arithmetic-data-types-in-c-6d31e82d5e85?source=collection_archive---------35----------------------->

# 有哪些不同的 C 数据类型？

C 标准将它的可用类型分成两大类*:对象类型和函数类型。*

*C 中的 ***对象类型*** 有:*

```
*Integer types such as int  , short  , enum  …
Floating point number types such as float  , double , long double .
Structures
Unions
Arrays
Pointers
Atomic
void*
```

*在 C 中，对象类型也有 ***多种概念划分*** 。例如，标量类别是由整数(如`int`)和浮点类型(如`float`)以及指针组成的类别。*

*一个关于算术类型之间转换的教程，可以在这里找到，这个 ***教程是关于*** 转换，非算术类型之间的转换。*

# *什么是虚空型？*

*void 类型 ***可以理解为*** *为*一个值的缺席。
***例如*** ，当声明一个函数有一个 void 返回类型时，这意味着这个函数没有返回值。此外，当声明该函数具有未命名的 void 参数类型时，这意味着该函数没有参数，因此它不带任何参数。*

```
*#include<stdio.h>void helloWorld(void ){
    printf("hello world \n" );
    /*The helloWorld function , has no parameters , 
      as such it has one parameter of type void . 
      It returns no values , as such as its 
      return type , it also has the void type .*/}int main(void ){
    helloWorld( );}
/*Output : 
hello world .*/*
```

*void 类型，也可用于 ***忽略任何值*** ，由任何表达式返回。因此，转换为 void 类型的表达式，其值会被忽略或丢弃。计算一个空表达式的副作用*

```
*(void) 1 ; 
/* The literal 1 is of the int type , 
   it is cast to the void type . The 
   result is an expression of the 
   void type , as a consequence , 
   the expression has no value .*/void helloWorld(void ){
    printf("hello world \n" );}helloWorld("Hello world" );
/*The expression , has a void type , 
  since the return type of the 
  helloWorld function is void . As
  a consequence , the expression has
  no value .  The  helloWorld function , 
  is used for printing the message 
  Hello world .*/*
```

*类型为 void， ***的表达式不能被*** 转换为任何其他类型。*

# *指针类型*

## *有哪些不同的可用指针类型？*

*`C` ***中的指针可以是指向*** *一个对象类型的指针，也可以是指向一个函数的指针。**

## *哪些整数类型能够存储指针？*

*一个 ***指针保存着*** 的一个地址。地址有数值，也有类型。*

```
*#include<stdio.h>int main(void ){
    int val_i = 1 ; 
    /*Declare and initialize the 
      variable val_i .*/int *ptr_i = &val_i;
    /*ptr_i is a pointer to an int .
     It contains a numeric value , 
     which is the address of 
     val_i . This address is of type 
     int . So when dereferencing 
     this address , a specific number of 
     bits , equal to the number of bits 
     in the type int , is read .*/printf("[@address](http://twitter.com/address) : %p -> value : %d\n", ptr_i , *ptr_i ) ;
    /*Print the address stored in ptr_i ,
      and dereference the address , 
      to get the stored value .*//*Output :
[@address](http://twitter.com/address) : 0x7fff59e438ec -> value : 1 */}*
```

*指针 中存储的 ***数值为整数数据类型。C 标准规定，存储在指针中的整数数据类型可以存储在`intptr_t`和`uintptr_t`中。因此，指针和`intptr_t`或`uintptr_t`之间的转换总是由 C 标准定义的。在`stdint.h`标题中定义了`intptr_t`和`uintptr_t`。****

*对于*其他整数数据类型，如果指针的整数值，大于要转换为的范围，整数数据类型，则转换的结果不定义。反之，如果整数数据类型值的位数大于`intptr_t`或`uintptr_t`，则转换结果未定义。**

```
**#include<stdio.h>
#include<stdint.h>int main(void ){int val_i = 1 ; 
  /*Declare an int variable 
    val_i , and  initialize 
    it , with 1 .*/int *ptr_i = &val_i;
  /*Declare a pointer to an integer , 
    and initialize it with the address
    of val_i */uintptr_t t_uip = (uintptr_t) ptr_i;
  /*Cast a pointer to uintptr_t */printf("ptr_i : %p , t_uip : %#lx\n" , ptr_i , t_uip );
  /*Print the address stored in the pointer
    ptr_i , and the hexadecimal numeric value
    stored in t_uip .*/ptr_i = (int * ) t_uip;
  /*Cast the integer value stored 
    in t_uip , to a pointer , to an 
    int ,  and assign the result
    to ptr_i .*/printf("ptr_i : %p , *ptr_i : %d\n" , ptr_i , *ptr_i );
  /*Print the address of ptr_i , and the 
      value stored by the address 
      referenced by ptr_i .*/
}/*Output : 
ptr_i : 0x7fff599d88ec , t_uip : 0x7fff599d88ec
ptr_i : 0x7fff599d88ec , *ptr_i : 1 */**
```

# **什么是指向 void 的指针？**

**一个指向 void 的指针，或者说 void 指针， ***可以理解为含义*** ，那对于现在来说，指针的类型，是不感兴趣的。在这种情况下，缺少值是地址的类型，而不是存储在地址中的数值。如前所述，指针变量存储地址。地址有一个数值和一个类型。**

```
**#include<stdio.h>int main(void ){
  void *ptr_v = (void *) 1 ;
  /*1 is an int literal , an integer
   can be cast to any pointer type .
   In this case , it is cast to 
   a pointer to void .
   The gotten address has a numeric value , 
   it has a void type , as such it does 
   not hava a type .*/
  printf("The stored address in ptr_v numeric value is : %p\n" , ptr_v );}/*Output : 
The stored address in ptr_v numeric value is : 0x1 .*/**
```

**任何 ***类型的指针都可以被转换成*** 类型的空指针，而任何空指针类型，都可以被转换成任何指针类型。**

```
**/*In this First example , any pointer sent to
  the printAddressVar function , is
  cast to a void pointer .
  The printAddressVar function , prints
  the numeric value of the address
  stored in the pointer .*/#include<stdio.h>void printAddressVar(void *var_vptr){
  printf("%p\n" , var_vptr );}int main(void ){
  int flowers_i = 10 ;
  printAddressVar(&flowers_i );
  float angle_f = 3.4f;
  printAddressVar(&angle_f );
  double elevation_d = 1.3;
  printAddressVar(&elevation_d );}
/*Output : 
0x7fff5dc4f8e4
0x7fff5dc4f8e0
0x7fff5dc4f8d8 *//*Second example .*/#include<stdio.h>int add (int x , int y ){return x+y; };
int sub (int x , int y ){return x-y; };
int mul (int x , int y ){return x*y; };
/*Declare , the add , sub , and multiply
  function .*/typedef int (* fct_signature ) (int , int) ;
/*fct_signature is alias to :  pointer to a
  function that takes two int , and
  returns , an int .*/int main(void ){
  void *arr_vptr[] = {add , sub , (void *) mul };
  /*add , sub , mul , are functions .
    A pointer to each , has a signature of
    int (* ) (int , int ) .
    Each pointer to each function is cast
    to the void pointer . 
    The gotten casts are stored in
    the array arr_vptr .*/int x = 1 , y = 2 , result = 0;result = ((fct_signature ) arr_vptr[0] )(x , y );
  /*Cast arr_vptr[0] to a pointer to a function .
    The pointer to the function has a signature :
    int (* ) (int , int ). After the casting ,
    the function is called .*/
  printf("%d\n" , result );
  /*Output : 3 */result = ((int (* ) (int , int )) arr_vptr[1] )(x , y );
  printf("%d\n" , result );
  /*Output : -1 */result = ((fct_signature ) arr_vptr[2] )(x , y );
  printf("%d\n" , result );
  /*Output : 2 */}**
```

## **什么是空指针？**

**空指针常量，或简称为空指针， ***由 C 标准的*** 定义，作为具有值的常量文字`0`。
空指针常量，也是由 C 标准定义的，将它的常量文字值`0`转换为指向 void 类型的指针。这两个定义是相等的，或者说是相同的。**

```
**#define NULL_PTR_CONSTANT 0#define NULL_PTR_CONSTANT ((void* ) 0 )**
```

**`*NULL*` ***宏，是标准头文件`stddef.h`中定义的*** ，为空指针常量。**

**一个 ***空指针常量可以被*** 强制转换为任何其他指针类型，结果是一个空指针，属于那种类型。所有空指针比较起来都是相等的。**

```
**#define NULL ((int * ) 0 )
/*Cast the null pointer 0 , to
  a pointer to an int , the 
  result is a null pointer .*/#include<stdio.h>int main(void ){
  printf("%d\n" , (int *) 0 == 0 ); 
  /*All null pointers are equal , printf 
    outputs 1 , which is true . 
    1 */}**
```

**空指针不等于非空指针，所以空指针不等于没有值`0`的指针。在指针变量的初始化中，可以使用空指针**

```
**#include<stdio.h>int main(void ){
    int var_i = 1 ;int *var2_ip = &var_i;
    /*Initialize var2_ip with the address 
      of var_i .*/int *var3_ip = 0 ;
    /* The null pointer constant 0
       is cast , to (int * ) , the
       result is  a null pointer of the 
       int type , the null pointer 
       is stored in var3_ip .*/

    printf("%d\n" , var3_ip != var2_ip );
    /*A null pointer is not equal , to a 
      not null pointer , as such , printf
      outputs 1 , which is true .*/}**
```

**空指针常量 ***可以理解为*** ，本身具有常量文字`0`的值，用来表示空指针的位模式不一定都是`0`。空指针常量表示地址中缺少数值，因此取消空指针的引用是未定义的。地址本身可以是无类型的，比如在`void`空指针的情况下，或者是有类型的，比如在`int`空指针的情况下。**

**将空指针常量 ***转换为整数类型*** 会产生`0`，因为空指针的值，而不是空指针解引用的值，是常量文字`0`。**

```
**#include<stdio.h>int main(void ){void *ptr_v = 0 ;
  printf("%d\n" , (int ) ptr_v );int *ptr_i = 0;
  printf("%d\n" , (int ) ptr_i ); }
/*Output :                                                                                                                  
0
0*/**
```

## **强制转换对象类型指针**

*****要记住*** 的关键一点，就是指针存储的是一个地址。地址有一个数值，因为 C 是一种类型化语言，所以地址有一个类型。
在处理对象类型指针之间的转换时，存储在地址中的数值不会改变，只有地址的类型会被重新解释。**

**话虽如此， ***对象类型指针，可以在*** 之间相互转换，唯一的要求就是对象类型有一个共同的对齐方式。对齐是对象在内存中可以放置的位置。如果对象类型没有公共的对齐方式，则将一个指针转换为另一个指针的结果是不定义的。**

```
**/*Example 1 */#include<stdio.h>int main(void ){

  float var_f = 1.0f ;
  int var_i = (int ) var_f ; 
  /*Casting a float type to an int 
    type , will change its bit 
    representation .
    1.0f is encoded as :   
      00111111100000000000000000000000
    1 is encoded as :
      00000000000000000000000000000001*/printf("%f , %d\n" , var_f , var_i );
  /*Output : 
    1.000000 , 1 */float *ptr_f = &var_f ;
  int *ptr_i = (int *) ptr_f ;
  /*ptr_f is cast to a pointer 
      to an int . The address 
      stored in ptr_f is copied to 
      ptr_i . 
    The bit pattern stored at that 
      address did not change . It is 
      00111111100000000000000000000000 */
    printf("%f , %d\n" , *ptr_f , *ptr_i );
    /*Output : 
      1.000000 , 1065353216 */ }/*Example 2 */#include<stddef.h>
#include<stdio.h>void toHex(unsigned char *ptr_uc , size_t size_data ){
    printf("%p : " , ptr_uc );
    /*Print the address of ptr_uc */
    for(size_t i = 0 ; i < size_data ; i++ )
        printf("%02x" , ptr_uc[i] );
        /*Print the hexadecimal representation 
          Of data .*/
    printf("\n" );}struct flag{
  unsigned char num_stars ;
  unsigned int num_colors;
};int main(void ){
  struct flag var_struct_flag = {255 , 4294967295 };
  toHex((unsigned char *)&var_struct_flag , sizeof(struct flag ));
  /*Output : 
    0x7fff539528e8 : ff000000ffffffff , 
      address         data hex dump
    The structure is padded with 6 bytes , 
    this is why , there are six 0 between
    255 , and 4294967295 .*/struct flag var_struct_flag1 = {0 , 0x0000FFFF };
  toHex((unsigned char *)&var_struct_flag1 , sizeof(struct flag ));}
  /*Output : 
    0x7fff599b68e0 : 00000000ffff0000 
       adress         data hex dump
    This is a little Indian machine , 
    since the int type is stored in
    reverse .*/**
```

*****如前所述*** ，空指针可以被转换成任何其他指针，反之亦然，而空常量指针，可以被转换成任何其他指针。**

## **合格和不合格指针类型之间的转换**

**C 标准定义，将一个指向非限定类型的指针，比如`int *`，强制转换为限定类型的指针，比如`const int *`、**、*总是定义为*、**。**

**C 语言中 ***可用的类型限定符*** 有`const`、`volatile`、`restrict`和`_Atomic`。**

```
**#include<stdio.h>int main(void ){
  int var_i = 1 ;
  int *ptr_i = &var_i ;
  const int *ptr_ci = ptr_i;
  /* *ptr_ci = 10;
     is illegal , because the
     pointer is a pointer to
     a const int. */
  *ptr_i = 10 ;
  /* Legal , because the pointer
     is a pointer to an int .*/}**
```

## **强制转换函数类型指针**

****、*的指针可以在*、**之间相互转换。此外，如前所述，void 类型可以转换为任何其他类型，任何其他类型都可以转换为 void 类型，空常量指针可以转换为任何其他指针类型。**

**如果指向一个函数的指针被强制转换为指向另一个函数、指针类型和目标函数类型，则 ***与*** 源函数类型不兼容，调用使用目标类型的函数，未定义。**

```
**int negate(int x ) {return -x ;}
int subtract(int x , int y ) {return x-y ;}int main(void ){
  int (* sig1_ptr ) (int )  = negate;
  int (* sig2_ptr ) (int , int ) = subtract;
  sig2_ptr = (int (* ) (int , int ))  sig1_ptr;
    /* Any function pointer can be cast
       to any other function pointer .
       If the signature of the casted 
       function , is not compatible with the  
       signature of the pointer function type , 
       calling the function pointed by the 
       pointer , is not defined . Hence 
       sig2_ptr(1 , 1 ) is not defined .*/ }**
```

## **比较指针的顺序**

*****关系运算符*** 、小于`<`、小于等于`<=`、大于`>`、大于等于`>=`，可以用来比较顺序。**

**比较指针的顺序， ***仅针对*** 对象类型指针定义，仅在比较指针与结构、数组和联合的成员时使用。**

**指向结构中的对象的指针，在同一结构中的对象之后声明，或者指向数组中的元素的指针，其位置比同一数组中的元素更高，顺序也更高。
反过来也是如此，所以指针 ***指向的对象在*** *一个*结构中，声明在同一个结构中的对象之前，或者到一个数组中的元素之前，有一个位置较低的，那么同一个数组中的元素，有一个较低的顺序。
***最后一个元素可比*** 为数组中的顺序，等于数组长度加一。**

**指向同一个 的任意 ***成员的指针，具有相同的顺序。指向相同数组成员或相同结构成员的指针，具有相同的指针地址:类型和数值，也具有相同的顺序。*****

```
**#include<stdio.h>struct rational {
  int numerator ;
  int denominator; };union search{
  int close_i ;
  int far_i; };int main(void ){struct rational var_rat = {1 , 2 };int *num_ip = &var_rat.numerator ;
  /*Store the address of numerator , in
    the pointer num_ip*/
  int *den_ip = &var_rat.denominator ;
  /*Store the address of denominator ,
    in the pointer den_ip */printf("%d\n" , num_ip < den_ip );
  /*numerator is declared before denominator ,
   in struct rational , as such its address
   has a lower order .
   Output : 1 .*/union search var_ser;int *close_ip = &var_ser.close_i;
  int *far_ip = &var_ser.far_i;printf("%d\n" , close_ip <= far_ip );
  /*close_i and far_i , are member of the
    same union , as such their address
    have the same order ,
    Output : 1 .*/
  printf("%d\n" , close_ip < far_ip );
  /*close_i and far_i , are member of the
    same union , as such their address
    have the same order ,
    Output : 0 .*/int arr_i[] = {1 };printf("%d\n" , arr_i < arr_i + 1 );
  /*address last element of array
    plus one , has higher order than
    preceding element .
    Output : 1 .*/
  printf("%d\n" , num_ip <= &var_rat.numerator );
  /*Pointers same member of structure ,  are
    equal .*/}**
```

## **比较指针是否相等**

**当使用等式`==`或差分运算符`!=`、**、*时，两个指针等于*、**，如果存储的地址具有相同的数值和相同的类型。**

**如果一个操作数是指针，第二个是空指针常量，那么在执行比较之前， ***空指针常量*** 被转换为指针类型。**

**如果一个操作数是指向对象类型的指针，而另一个操作数是指向 void 的合格或不合格 ***指针，那么对象指针就被强制转换为指向 void 的合格或不合格指针。*****

```
**#include<stdio.h>int main(void ){
  int var_i = 1;
  int *ptr_i = &var_i;
  int *ptr1_i = &var_i;double var_d = 1.0;
  void *ptr_v = &var_d;printf("%d\n" , ptr_i == ptr1_i );
  /*Both ptr_i , and ptr1_i , address 
    contain the same numeric value , 
    and type , they are equal
    Output : 0.*/printf("%d\n" , ptr_i == 0 );
  /*The constant pointer literal
    0 , is cast to a null pointer
    literal of the type int . 
    A null pointer is not equal 
    to a non null pointer , as such
    the result is false .
    Output : 0.*/printf("%d\n" , ptr_i == ptr_v );
  /*ptr_v is a void pointer , ptr_i
    is cast to a void pointer , the
    numeric value of the address are
    not equal , as such the result
    is false .
    Output : 0 .*/ }**
```

# **逻辑运算符**

**逻辑运算符，and `&&`，or `||`，not `!`，**，*可以与*，**，*一起用于*标量类型。标量类型是整数类型、浮点类型和指针类型。**

**如果`&&`的任何一个操作数是`0`，它将产生`0`。**

**`||`将产生`1`，如果它的任何一个操作数是`1`。**

**如果其操作数为`0`，则`!`将产生`1`，否则产生`1`。**

**所以 ***在指针的情况下，如果*** 指针是空指针，那么`&&`将产生假或`0`，如果指针不是空指针，那么`||`将产生真，或`1`。至于`!`，如果指针不是空指针，它将产生`0`，如果指针是空指针，它将产生`1`。**

```
**#include<stdio.h>int main(void ){
  int var_i = 1 ;
  int *ptr_i = 0;
  if(!ptr_i )
    /* Apply , the not operator ,
      on ptr_i . ptr_i is the null
      pointer , it has a value of 0 ,
      as such after applying the not
      operator , it will have a value of
      1 . When 1 , if execute , the
      following statement , which initialize
      the pointer to the address of
      var_i .*/
    ptr_i = &var_i;
  if(ptr_i || 0 )
    /* ptr_i , is not null , as such
      || does not evaluate the ,
      second expression , and returns
      1\. On 1 , if exceutes the
      following statement , which
      prints the value found , 
      in ptr_i .
      Output : 1 .*/
    printf("%d\n", *ptr_i );
  if(ptr_i && 0 )
    /* ptr_i is not a null pointer ,
      && evaluates 0 . On 0 ,
      it returns 0 .
      If , on 0 , does not execute , 
      the next statement , hence
      printf is not executed .*/
    printf("%d\n", *ptr_i );}**
```

## **添加**

**对于指针类型，使用加法运算符，`+`，**时定义，*，一个操作数是整数类型，第二个操作数是指向完整对象类型的指针。*****

****一个 ***完整的对象类型*** ，是一个有大小的对象。例如，void 类型不是一个完整的对象类型，因为它没有大小，一个已声明但未定义的结构，因此它没有主体，这是不完整类型的另一个示例。****

```
****struct t_s;****
```

****当将一个整数值加到存储在指针中的数值上时，加到 上的不是整数值 ***，而是指针指向的乘以对象大小的整数值。*******

****对于指针的整数加法， ***有效*** ，指针必须是指向数组对象成员的指针，或者是指向数组最后一个对象成员之后的元素的指针，加法的结果必须是指向数组对象的指针，或者是指向数组最后一个对象之后的元素的指针，否则不定义指针加法。****

****数组最后一个对象成员之后的元素是 ***不一定是空的*** 指针，但是在所有情况下都不能使用`*`操作符取消引用。****

```
****#include<stdio.h>int main(void ){float arr_f[] = {1.0f , 2.0f };
  float *ptr_f = &arr_f[0 ];printf("sizeof(float ) : %zd\n\n" , sizeof(float ));
  /*Print the size of the type of the object
    pointed by ptr_f */printf(" %p : ptr_f\n %p : ptr_f+0\n %p : ptr_f+1\n %p : ptr_f+2 \n" ,
                ptr_f , ptr_f + 0 , ptr_f + 1 , ptr_f + 2  );
  /*Perform pointer addition , and print the
   successive addresses .
   add 0 , 1 , and 2 , to ptr_f . 
   ptr_f points to the first element ,
   of the array arr_f . The addition , 
   is defined , as long as
   the gotten pointer , is a pointer to an
   element of the array arr_f , or one past ,
   the last element , of the array arr_f .*/printf("ptr_f + 2 == (void * ) 0 -? %d \n\n\n" , (ptr_f + 2 )  == 0 );
  /*The element past the last object ,
    member of an array , is not
    necessarily the null pointer ,
    and it must not be dereferenced  .*/char *ptr_c = (char * ) ptr_f;
  /*Cast the pointer ptr_f , to a
    pointer to a char .*/printf("sizeof(char ) : %zd\n\n" , sizeof(char ));
  /*Print the size of the type
    of the object , pointed by
    ptr_c .*/printf(" %p : ptr_c\n %p : ptr_c+0\n %p : ptr_c+1\n %p : ptr_c+2 \n"
         " %p : ptr_c+3\n %p : ptr_c+4\n %p : ptr_c+5 \n"
         " %p : ptr_f+6\n %p : ptr_c+7\n %p : ptr_c+8  \n\n\n"
         , ptr_c , ptr_c + 0 , ptr_c + 1 , ptr_c + 2 , ptr_c + 3 , ptr_c + 4
         , ptr_c + 5 , ptr_c + 6 , ptr_c + 7 , ptr_c + 8 );
  /*Perform pointer addition . The pointer is now a pointer
    to an object of type char . The addition is still valid ,
    as long as the gotten pointer , is a pointer to
    an object in the array , or one past the last
    oject in the array . The array is now interpreted ,
    as being , an array of characters .*/}/*Output : 
sizeof(float ) : 40x7fff5230c8c0 : ptr_f
 0x7fff5230c8c0 : ptr_f+0
 0x7fff5230c8c4 : ptr_f+1
 0x7fff5230c8c8 : ptr_f+2 
ptr_f + 2 == (void * ) 0 -? 0sizeof(char ) : 10x7fff5230c8c0 : ptr_c
 0x7fff5230c8c0 : ptr_c+0
 0x7fff5230c8c1 : ptr_c+1
 0x7fff5230c8c2 : ptr_c+2 
 0x7fff5230c8c3 : ptr_c+3
 0x7fff5230c8c4 : ptr_c+4
 0x7fff5230c8c5 : ptr_c+5 
 0x7fff5230c8c6 : ptr_f+6
 0x7fff5230c8c7 : ptr_c+7
 0x7fff5230c8c8 : ptr_c+8 */****
```

## ****减法****

****减法*是* ***为指针*** 定义的，当进行减法时，是将一个指针减法到一个完整的对象类型，再从另一个指针减法到一个完整的对象类型。当从整数值中减去指向完整对象类型的指针时，也为指针定义了减法。一个完整的对象类型是有大小的。****

****为了使两个指针 ***相减有效*** ，它们必须指向属于同一个数组的对象。指向元素，越过属于数组的最后一个对象，也是允许的。****

****从另一个指针中减去一个指针的结果是类型`ptrdiff_t` 。这个结果 ***是两个指针之间的距离*** ，作为这两个指针的对象类型的大小的计数。`ptrdiff_t`被定义，在`stddef.h`头中。****

```
****#include<stdio.h>int main(void ){int arr_i [] = {0 , 1 };
  int *ptr_i = &arr_i[0 ];printf("%td\n" , ptr_i - &arr_i[1 ]);
  /*Print the difference between
    the two pointers . The result is
    of the type ptrdiff_t , hence the use
    of %td .
    Output : -1 .*/printf("%td\n" ,  &arr_i[2] - ptr_i );
  /*Print the difference between
    the two pointers . The distance from
    one past the last object , member of
    the array , to the first object member
    in the array is 2 int , since the pointers
    are of type int .
    Output : 2 .*/char *ptr_c = (char * )ptr_i ;
  printf("%td\n" ,  (char * ) &arr_i[2] - ptr_c );
  /*Cast ptr_i , to a pointer to a char .
    Print the difference , between one
    past the last object member of the
    array , now interpreted as a char ,
    and the first object member of the
    array .
    Output : 8 .*/}****
```

****从整数中减去一个指针， ***有效的*** ，指针必须指向数组中的一个对象，或者指向数组中最后一个对象成员之后的一个元素，结果必须指向数组中的一个对象，或者指向数组中最后一个对象之后的一个元素。****

*******从指针中减去一个整数*** 就是从指针中存储的地址的数值中减去指针所指向的整数，乘以对象的大小。****

```
****#include<stdio.h>int main(void ){unsigned char arr_uc[] = {0 , 255 , 0 , 255 };
  /*Declare an array of type unsigned char , 
    and initialize it .*/unsigned char  *ptr_uc = (unsigned char * ) &arr_uc[4 ];
  /*Get a pointer , to one past the last
    object  , member of the array 
    arr_uc .*/int length_arr_uc = sizeof(arr_uc ) / sizeof (unsigned char );
  /*Calculate the length of the 
    array of type , unsigned char .*/for(int i = length_arr_uc ; i >= 0 ; i-- ){
  /*Print the address , and value if any , 
    of address accessible using subtraction
    by an integer from a pointer , which points
    to one past the last object , member
    of the array .*/
    printf("%p : " , ptr_uc - i );
  if(ptr_uc - i != ptr_uc )
    printf("%u" , *(ptr_uc -i ));
  printf("\n" );}}/*Output : 
0x7fff559338e8 : 0
0x7fff559338e9 : 255
0x7fff559338ea : 0
0x7fff559338eb : 255
0x7fff559338ec :     */****
```

****当对指向完整对象类型的指针执行加法和减法时，指向不是成员 的完整对象 ***的指针，或者数组最后一个元素之后的指针，就好像该对象是长度为 1 的数组，并且是指针所指向的对象的类型。*******

```
****#include<stdio.h>int main(void ){int var_i = 2002874948 ;
  /*Declare an int variable ,
    having a value of 2002874948 .*/int size_of_var_i = sizeof(var_i );
  /*Get the size of the int 
    variable . The size is 
    returned in bytes . 
    1 char , has a size of 1 byte .*/char *ptr_c = (char *) &var_i;
  /*Get a pointer to the int 
    variable , and cast it to
    pointer , to a char .*/for(int i = 0 ; i < size_of_var_i ;  i++ )
    printf("%c" , *(ptr_c + i ));}
/*output :
Draw */****
```

## ****后缀和前缀、递增和递减运算符****

****后缀、前缀、递增和递减运算符:`++`、`--`、**、*可以在指针类型上使用*、**。****

****当递增或递减一个指针时，指针中存储的地址，**递增或递减了指针所指向的对象的大小*。*****

****使用后缀运算符递增和递减的结果是 ***只能从下一条语句*** 中访问，因此在执行它们的语句中，后缀递增和递减运算符返回操作数值不变。****

****前缀，递增和递减运算符，递增或递减操作数， ***返回递增或递减的*** 操作数。****

```
****#include<stdio.h>int main(void ){
  char array_c[ ] = "aa";
  /*Create an array of length 3 ,
    initialized with the characters
    a a , and terminated  with
    the null character . The null
    character , has all of its
    bits , set to 0 .*/char *ptr_c = &array_c[0 ];
  /*ptr_c is a pointer of
   typ char , it contains ,
   the address of the first
   object , member of the array
   array_c .*/while(ptr_c && *ptr_c != '\0' ){
    printf("%#2x\n" , *ptr_c++);}
  /*Print the hexadecimal representation ,
    of the data stored in array_c .
    The pointer is incremented using ,
    the postfix operator , hence 
    ptr_c value is only incremented , 
    starting the next statement . 
    Output :
    0x61
    0x61 */
  printf("\n" );ptr_c = &array_c[0 ];
  /*Rewind the pointer , 
    to the address of the 
    first object , stored in 
    array_c .*/while(ptr_c && *ptr_c != '\0' ){
    printf("%#2x\n" , *++ptr_c);}
  /*ptr_c contains the address 
    of the first object , member
    of the array array_c . 
    The while statement , loops
    through array_c elements , using
    the prefix increment operator , 
    the address stored in ptr_c , is
    first incremented . Next , it is
    dereferenced . 
    Output : 
    0x61
    0 */ }****
```

## ****乘法、除法****

****乘法和除法 ***只适用于*** 整数和浮点类型，因此不适用于指针类型。****

## ****位运算****

*******位运算只适用于*** 整数类型，因此它们不是为指针类型定义的。****

## ****条件运算符****

****条件运算符 ***的格式为*** :****

```
****Operand_One?Operand_Two:Operand_Three****
```

*******操作数 _One 必须是*** 标量。C #中的标量类型有整型、浮点型和指针型。****

****如果将 ***指针用作第二个或第三个操作数*** ，则:****

****要么，一个操作数是指向一个对象的指针，第二个操作数是指向一个 合格或不合格 void 指针的 ***。在这种情况下，指向对象的指针被转换为 void 指针的限定或非限定版本。*******

```
****int *ptr_i ;
const void *ptr_v ;
ptr_v = 1 ? ptr_i : ptr_v ;
/*When the first operand is  1 ,
  the second operand is evaluated .
  ptr_i is cast using 
  (const void * ) .*/****
```

****要么，一个操作数 ***是指针*** ***指向*** a C 类型，第二个操作数是空指针常量。在这种情况下，空指针常量被转换为所指向的 C 类型的指针。****

```
****volatile short *ptr_vs; 
ptr_vs  = 0 ? ptr_vs : 0 ;
/*When the first operand is 0 , 
  the third operand is evaluated.
  The result is a volatile
  short null pointer . */****
```

****要么，两个指针操作数，**、*都有相同的*、** C 类型，但有不同或相同的类型限定符。结果是一个指针，指向相同的类型，有所有的限定符。****

```
****volatile int *ptr_vi;
const int * ptr_ci;
const volatile int *ptr_cvi = 1 ? ptr_vi : ptr_ci ;
/*When the first operand is 1 , 
  the second operand ptr_vi is 
  evaluated.
  The result is a constant volatile 
  int pointer . */****
```

# ****数组****

****数组是一个聚合类型，不能将 ***转换为*** 之类的`(int [ ])`或`(int [3 ])`数组类型。只能对标量类型进行强制转换，如`(int * )` 。标量类型有整数类型、浮点类型和指针类型。****

*******按位运算符*** 如 shifting】，不适用于数组，它们只适用于整数类型。****

*******乘除*** ，只适用于整数和浮点类型，同样它们不适用于数组。****

****当一个数组 ***被传递给一个函数时，当执行相等比较(如`!=`)或顺序比较(如`<=`)时，该数组充当指针*** 。当执行逻辑运算时，如`&&`和执行加法、`+`和减法`-`，数组也被视为指针。****

****数组 充当的 ***指针，是数组第一个元素的地址，这是一个常量指针，所以它的地址不能改变。*******

```
****#include<stdio.h>int add(int *ptr_i , int length ){
  int sum = 0 ;
  for(int i = 0 ; i < length ; i++ )
    sum += *(ptr_i + i );
  return sum;}int main(void ){
  int arr_i[] = {1 , 2 , 4 , 6};
  /*Create an int array , containing
    4 elements .*/printf("sizeof(arr_i ) : %zd bytes\n", sizeof(arr_i ));
  /*Print the size of the array ,
    Output :
    sizeof(arr_i ) : 16 bytes */const int *cptr_i = arr_i;
  /*When assigning an array to a pointer ,
    the array act as a constant pointer to
    its first element .*/printf("sizeof(cptr_i ) : %zd bytes\n", sizeof(cptr_i ));
  /*Prints the size of the pointer , not the size
    of the array .
    Output :
    sizeof(cptr_i ) : 8 bytes */printf("Number of elements in array : %zd\n" , sizeof(arr_i ) / sizeof(arr_i[1 ] ));
  /*Print the number of elements in the array .
    This can be gotten by dividing the size of the
    array , which is 16 bytes , by the size of
    an element in an array. 
    The size of an int on this machine is 4 bytes ,
    as such the number of elements is 4 .
    Output :
    Number of elements in array : 4 */int * ptr_i = 0;
  if(ptr_i != arr_i )
    ptr_i = arr_i;
  /*When performing comparison
    operations , the array
    acts as a constant pointer , to
    its first element .
    The null pointer is different
    from a not null pointer , hence
    the assignment operation is 
    performed , and ptr_i is a 
    pointer , to the first
    element of the array .*/printf("2nd element of array is : %d\n" , *(arr_i + 3 ));
  /*Print the value of the second ,
   element of the array . When performing
   addition or subtraction on an array ,
   it acts as constant pointer to its
   first element . This is
   pointer addition or subtraction .
   Output :
   2nd element of array is : 6 */printf("sum elements array : %d\n" , add(arr_i , 4 ));
  /*When passed to a function , the array acts
    as a constant pointer , to its first element .
    The recieving function parameter , get the
    refered address .
    The add function calculates , the sum of the
    elements , of the array .
    Output :
    sum elements array : 13 */if( (arr_i < arr_i + 1 ) && arr_i )
    printf("True\n" );
  /*When using order comparison < <= >= > ,
    the array acts as a constant pointer ,
    to its first element , the address
    of the first element is less than
    the address of the second element
    of the array .
    && evaluates its first operand ,
    which returns 1 . Since the first
    operand returned 1 , && 
    evaluates , its second operand . 
    The second operand is arr_i  , since
    this is a logical operation , arr_i
    acts as a constant pointer to its
    first element , the gotten pointer 
    is not a null pointer , hence the 
    second operand of && evaluates to 1 ,
    as such && evaluates to 1 .
    When 1 , the if statement ,
    executes , the next statement ,
    which prints True .
    Output:
    True .*/}****
```

****使用后缀和前缀递增`++`和递减`--`运算符，数组 ***不能递增或递减*** 。****

# ****结构、联合****

****强制转换只适用于标量类型，标量类型有整数类型、浮点类型和指针类型，因此 ***不适用于*** 结构和联合。****

*******不能使用*** 顺序、或等式、或按位、或逻辑、或乘法和除法、或加法和减法、或后缀和前缀递增和递减运算符，以及结构和联合。****

****结构和联合 ***可以是三元运算符`?:`的第二个*** 和第三个操作数，在这种情况下，它们必须具有相同的类型。****

*****原载于 2020 年 12 月 29 日 https://twiserandom.com*[](https://twiserandom.com/c/conversion-and-operations-on-the-non-arithmetic-data-types-in-c/)**。******