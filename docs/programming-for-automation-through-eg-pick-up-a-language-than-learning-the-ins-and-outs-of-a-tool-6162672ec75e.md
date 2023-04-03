# 通过学习一门语言，然后学习一个工具的细节，为自动化编程。

> 原文：<https://medium.com/analytics-vidhya/programming-for-automation-through-eg-pick-up-a-language-than-learning-the-ins-and-outs-of-a-tool-6162672ec75e?source=collection_archive---------17----------------------->

要想在测试自动化领域脱颖而出，依靠单一的测试工具是不可取的。不管你必须掌握怎样编码的工具。学习一门编程语言，然后学习自动化工具的细节会给你带来优势。在这篇文章中，比较了使用相同库执行相同操作的两个脚本。以下截图描述了该功能。机器人框架是用于该示例的工具，但是无论使用何种工具，类似的特征都是可用的。

![](img/363c3194eee1c2b2f6ade3fc4e0ef9e1.png)

[https://www . Samsung . com/uk/support/TV-audio-video/what-the-default-pin-of-my-television/](https://www.samsung.com/uk/support/tv-audio-video/what-is-the-default-pin-of-my-television/)

**剧本 1 :**

```
*****Settings*****
Library String
Library SeleniumLibrary
Library Collections
```

机器人框架测试功能正在使用*字符串*、*硒库*和*集合*库为该脚本进行扩展。

```
*****Variables*****
${PIN_LOCATOR1}    id=PageConfiguration.PIN1
${PIN_LOCATOR2}    id=PageConfiguration.PIN2
${PIN_LOCATOR3}    id=PageConfiguration.PIN3
${PIN_LOCATOR4}    id=PageConfiguration.PIN4
${CONFIRM_PIN_LOCATOR1}    id=PageConfiguration.CONFIRM_PIN1
${CONFIRM_PIN_LOCATOR2}    id=PageConfiguration.CONFIRM_PIN2
${CONFIRM_PIN_LOCATOR3}    id=PageConfiguration.CONFIRM_PIN3
${CONFIRM_PIN_LOCATOR4}    id=PageConfiguration.CONFIRM_PIN4
@{PIN_NUMBER}    null
```

标量变量用于存储定位器，因此变量名按原样替换为其字符串值。还创建了一个列表变量，并将其初始化为 null。Robot Framework 将其变量存储在一个内存中，并允许将它们用作标量或列表。使用变量作为列表要求其值是列表，不允许字符串作为列表，但接受其他迭代对象。

```
*****Keywords*****
Generate Enter And Confirm A Random PIN 
    ${PIN_NUMBER}   Generate A Random Personal Identification Number
    Enter The Personal Identification Number     ${PIN_NUMBER}
    Confirm The Personal Identification Number   ${PIN_NUMBER}
```

这个高级关键字协作生成、输入和确认个人标识号。生成一个随机的四位数个人标识号，并将其存储在列表变量中。从列表变量中提取个人标识，并在相应的文本框中输入值。将从列表中再次提取的值输入到相应的文本框中，以确认个人识别号。

```
Generate A Random Personal Identification Number
    ${number1}    Generate Random String length=1 chars=123456789
    Append To List    ${PIN_NUMBER}    ${number1}
    ${number2}    Generate Random String length=1 chars=123456789
    Append To List    ${PIN_NUMBER}    ${number2}
    ${number3}    Generate Random String length=1 chars=123456789
    Append To List    ${PIN_NUMBER}    ${number3}
    ${number4}    Generate Random String length=1 chars=123456789
    Append To List    ${PIN_NUMBER}    ${number4}
    [Return] ${PIN_NUMBER}
```

这个低级关键字生成随机的个人标识号，并以列表的形式返回值。关键字'*生成随机字符串*'创建个人 PIN 字符，该关键字从*字符串*库中扩展而来。生成随机字符串要考虑的长度和字符是通过参数定义的。生成的每个新字符都附加在列表的底部，关键字从*集合*库中扩展而来。

```
Enter The Personal Identification Number
    [Arguments]   ${PIN_NUMBER}
    Input Text    ${PIN_LOCATOR1}    ${PIN_NUMBER[0]}
    Input Text    ${PIN_LOCATOR2}    ${PIN_NUMBER[1]}
    Input Text    ${PIN_LOCATOR3}    ${PIN_NUMBER[2]}
    Input Text    ${PIN_LOCATOR4}    ${PIN_NUMBER[3]}
```

该低级关键字输入 PIN，该关键字从 *SeleniumLibrary* 扩展而来。存储在标量变量中的定位器是元素识别策略。因为 PIN 存储在列表变量中，所以列表变量的索引用于访问要输入的数据。

```
Confirm The Personal Identification Number
    [Arguments]   ${PIN_NUMBER}
    Input Text    ${CONFIRM_PIN_LOCATOR1}    ${PIN_NUMBER[0]}
    Input Text    ${CONFIRM_PIN_LOCATOR2}    ${PIN_NUMBER[1]}
    Input Text    ${CONFIRM_PIN_LOCATOR3}    ${PIN_NUMBER[2]}
    Input Text    ${CONFIRM_PIN_LOCATOR4}    ${PIN_NUMBER[3]}
```

这个低级关键字再次输入 PIN 以确认新值。该关键字的行为与前任相同，唯一的区别是标量变量。

**精炼剧本 1:**

```
*****Settings*****
Library String
Library SeleniumLibrary
Library Collections
```

这些库类似于前面的脚本。

```
*****Variables*****
${PIN_LOCATOR}    id=PageConfiguration.PIN
${CONFIRM_PIN_LOCATOR}    id=PageConfiguration.CONFIRM_PIN
@{PIN_NUMBER}    null
${RETRY_ATTEMPTS}    5x
${RETRY_AFTER}    5000ms
```

只有一个标量变量用于捕捉位置。字符串末尾用于标识文本框的唯一数字不作为标识符的一部分。类似的策略适用于创建用于确认 PIN 的文本框的标识符。和前面的脚本一样，创建了一个列表变量并初始化为零。引入了两个新的标量变量来定义重试尝试和尝试之间的间隔。

```
*****Keywords*****
Generate Enter And Confirm A Random PIN 
    ${PIN_NUMBER}   Generate A Random Personal Identification Number
    Enter The Personal Identification Number    ${PIN_LOCATOR} ${PIN_NUMBER}
    Enter The Personal Identification Number   ${CONFIRM_PIN_LOCATOR} ${PIN_NUMBER}
```

高级关键字的目的保持不变，但是修改了低级关键字以适应改进的方法。

```
Generate A Random Personal Identification Number
     :FOR     ${index}    IN RANGE    1    5
     \   ${number}   Generate Random String length=1 chars=123456789
     \   Append To List    ${PIN_NUMBER}    ${number}
     [Return]    ${PIN_NUMBER}
```

这个低级关键字还生成随机的个人标识号，并以列表的形式返回值。这个关键字使用了四次循环，而不是生成和追加字符。机器人框架中使用的 for 循环语法是使用内置的 *range()* 函数从类似的 Python 习语中派生出来的。循环开始引用起始限制定义并迭代，但不包括上限。默认情况下，这些值以 1 为增量，也可以给出指定要使用的增量的步长值。如果步长为负，也可以减少。

```
Enter The Personal Identification Number
    [Arguments]    ${locator}    ${PIN_NUMBER}
    :FOR    ${index}    IN RANGE    1    5
    \ ${locator} = catenate    ${locator}    ${index}
    \ Keyword For Identifying And Entering   ${locator} ${PIN_NUMBER[${index}]}
    \ ${locator}    Get Substring    ${locator}    0    -1
```

这个低级关键字也输入 PIN。该关键字动态创建标识符，而不是使用四个静态标识符。在每次迭代之后，使用从*字符串*库中扩展的‘获取子串’关键字剥离附加的字符。子字符串提取使用定义的起始和结束索引。开始索引是包含性的，但结束索引是排他性的。索引从零开始，负索引从末尾开始引用字符。使用相同的关键字重新输入 PIN，但使用另一个标量变量。

```
Keyword For Identifying And Entering
    [Arguments]    ${locator}    ${text}
    Wait Until Keyword Succeeds   ${RETRY_ATTEMPTS}  ${RETRY_AFTER}     
    Identifying And Entering    ${locator}    ${text}
```

这个常用关键字用于根据传递的参数向文本框输入值。内置的“*等待关键字成功*”关键字运行指定的关键字，并在失败时重试。这些参数定义了重试次数以及在前一次尝试失败后再次尝试之前等待的时间。这个简单而强大的关键字可以防止最低级别的剥落。

```
Identifying And Entering
    [Arguments]    ${locator}    ${text}
    Element Should Be Enabled    ${locator}
    Input Text    ${locator}    ${text}
```

该关键字确保操作在适当的时间发生。在这种情况下，在文本框中输入值之前，验证文本框是否已启用。

 [## 机器人框架

### 用于验收测试和 ATDD 的通用测试自动化框架

robotframework.org](https://robotframework.org/#/)  [## 机器人框架/硒库

### SeleniumLibrary 是一个机器人框架的 web 测试库，它在内部使用 Selenium 工具。该项目是…

github.com](https://github.com/robotframework/SeleniumLibrary/)