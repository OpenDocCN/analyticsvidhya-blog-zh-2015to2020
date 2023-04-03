# 可以用 Python 中的语音机器人存储员工信息吗？

> 原文：<https://medium.com/analytics-vidhya/can-you-store-employee-information-using-a-voice-bot-713c1d9a770f?source=collection_archive---------31----------------------->

![](img/cae289d796bf34aa08d756bcc9409375.png)

本·怀特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

绝对的！

最近，我用 Python 创建了一个语音机器人来存储基本的员工信息，例如员工 id、员工的名和姓以及员工的工资。它不仅使用语音机器人来存储基本的员工信息，还允许你手动输入员工信息。这里是[的源代码](https://github.com/hassanAli123-stack/SQLITE3-project)。

然而，在开始这个项目之前，您需要一些工具。其中最重要的工具是 Python。

# **如何入门？**

该项目要求您对 Python 中的 SQLITE3 模块有一个基本的了解，因为您将使用该模块来创建数据库。您还需要对 Python 中的类和实例有一个基本的了解。

为了理解 Python 中的类和实例，你可以浏览科里·斯查费的第一系列视频“类和实例”。

对于 SQL 语法的基本理解，你可以跟随 [Mike Dane 的课程](https://www.mikedane.com/databases/sql/)学习 SQL。然而，对于这个项目，您只需要知道如何使用 SQL 更新、创建或删除表。

熟悉了 SQL 语法之后，你就可以开始阅读科里·斯查费的 SQLITE3 教程了。

如果您可以使用 Python 中的类和实例，并且知道如何使用 SQLITE3 模块，那么就不要再等了，开始吧。

**开始使用前的快速注意事项**:如果您看到奇怪的函数名，请不要混淆。我已经详细描述了每个函数的功能。

# **第一步:创建雇员类**

employee 类将用于创建雇员的名、姓和工资。这个类只有几行代码，可以在项目本身中编写，但是，在不同的 python 文件中创建 employee 类是我开始 OOP 编程和理解如何将类导入另一个 python 文件的一种方式。

```
class Employees:
    *"""

    This class saves the employee first name, last name, and pay. It's imported in the project SQLITE3

    """* def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
```

在这里，您创建了名为`Employees`的类。在 Employee 类中，您将创建具有参数`self, first, last, pay`的构造函数方法。`self`参数指的是类 Employee 的实例，而其他参数`first, last, pay`用于初始化类实例的数据属性。在构造函数方法内部，当创建一个类的实例时，您将把`first, last, pay`的值设置为在构造函数方法中传递的值。例如，参数`first`被设置为使用类实例`self.first`在构造函数方法中传递的值。

上课到此为止。现在，让我们继续创建实际的项目…

# **第二步:创建项目**

## 要为项目导入的模块:

```
# modules for the voice bot
import speech_recognition as sr
from gtts import gTTS
import playsound# modules for the database
import random
from employee_class import Employees
import sqlite3
import os
import re
```

## 创建语音机器人:

```
def speak_text(speak):
 *“””
 This function makes the voice bot speak a specific command.
 “””* rand = random.randint(1, 10000)
 filename = ‘file’ + str(rand) + ‘.mp3’
 tts = gTTS(text=speak, lang=’en’)
 tts.save(filename)
 playsound.playsound(filename)
 os.remove(filename) 
```

`speak_text()` 函数创建一个 mp3 文件，语音机器人通过该文件说话。每次程序运行时，变量`filename`都会创建一个新的 mp3 文件，因为如果语音机器人通过同一个 mp3 文件说话，程序会给出一个错误。`tts`变量使用了`gTTS` (Google-Text-To-Speech)模块。`gTTS`模块说出字符串`text`，该字符串被设置为等于函数参数`speak`中传递的字符串。对于我们的项目来说，`gTTS`模块使用`lang=’en` 来说英语，然而，该模块支持多种语言。因为`tts`变量使用 Google API 将字符串转换成音频，所以您将使用`tts.save`将音频保存在 mp3 文件`filename`中。最后，因为您不希望随机文件保存在您的文件夹中，所以在语音机器人说话后，您使用`os.remove(filename)`方法删除文件名。

## 创建函数 get_audio():

```
def get_audio():
 "*""
This function takes input from the user through the microphone and returns an exception if the command is not understood by the voice bot*""" r = sr.Recognizer()
 with sr.Microphone() as source:
 audio = r.listen(source)
 said = ‘’try:
 said = r.recognize_google(audio)
 print(said)except Exception as e:
 print(‘Exception’ + str(e))
 return saidspeak_text(‘welcome to my SQLITE project that stores employee information into a database.’)
speak_text(‘you can enter the data manually or use the voice bot’)`
```

get_audio()函数通过麦克风接受用户的语音命令。首先，创建变量`r`来从 speech_recognition 模块创建一个识别对象。在语句`with sr.Microphone()as source` 中，您创建了`audio`变量，它监听来自麦克风的音频`source`。在`try`语句中，您使用 Google API `r.recognize_google (audio)`来识别音频。当 Google API 不能识别用户所说的内容时，它会返回一个异常。

最后，使用`speak_text()` 函数向用户介绍项目。

## 创建数据库:

```
conn = sqlite3.connect('employee.db')

c = conn.cursor()

'''  Create a table called 'employees'  using SQLITE3 to store data'''

c.execute(""" CREATE TABLE IF NOT EXISTS employees (
              employee_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
              first TEXT,
              last TEXT,
              pay INTEGER
            )""")

conn.commit()
```

为了创建数据库，您创建了变量`conn` ，它将数据库保存为`employee.db`。变量`c`被设置为等于`conn.cursor()`。cursor()方法允许您编写 SQL 命令。之后，您可以使用`c.execute()`中的`execute` 方法来创建 employees 表。该表存储了雇员 id、名字、姓氏和工资。列`employee_id`用于自动为员工分配一个代理键，以便用户可以区分同名的员工。在命名每一列之后，您分配每一列将要保存的数据类型。最后，使用`conn.commit()`方法保存对表所做的更改。

## 创建一组在表中插入、删除或获取雇员的函数:

为了对 employee 表进行更改并使事情变得简单，您必须创建插入雇员、删除雇员或从表中获取每个雇员信息的函数。

假设您知道基本的 SQLITE3 命令，这应该不会太复杂而难以理解。

**插入员工功能:**

```
def insert_emp(emp): 
“”” 
This function inserts the employee data into the table
 “”” 
with conn:
 c.execute(“”” INSERT INTO employees VALUES (?,?,?,?) “””, (None,   emp.first, emp.last, emp.pay)) 
conn.commit()
```

函数`insert_emp`将`emp`作为其参数。因为您正在将一个雇员插入到数据中，所以您将使用上下文管理器`with conn` ，其中`conn`是您保存文件`employee.db`的变量。`with` 是一个上下文管理器，它打开数据库，向其中写入一些数据，然后在数据更新时关闭数据库。上下文管理器很有用，因为如果你忘记关闭数据库，程序就会出错。在`c.execute`方法中，SQL 命令`INSERT INTO employees VALUES`将元组值`None, emp.first, emp.last, emp.pay`插入到表中。元组值`emp.first, emp.last, emp.pay`是 employee 类的实例，而`None`是自动递增的`employee_id`。为了保存修改，你会在最后写`conn.commit()`。

**删除员工功能:**

```
def delete_emp(id_emp):   
 """    This function deletes the employee from the data    """    with conn:       
 c.execute(""" DELETE FROM employees WHERE employee_id=:employee_id""",
 {'employee_id': id_emp})conn.commit()
```

函数`delete_emp`接受参数`id_emp`。由于雇员 id 对每个雇员都是唯一的，通过雇员 id 删除雇员信息将确保用户不会删除同名的雇员。就像`insert_emp`函数一样，您使用上下文管理器`conn`和方法`c.execute()`。然而，这一次您使用 SQL 命令`DELETE FROM employees`来删除雇员的 id，其中表中的`employee_id`等于用户在参数`id_emp`中传递的参数。要保存所有更改，您将使用`conn.commit()`。

**获取所有员工信息的函数:**

```
def get_employees():  
# gets all employee names   
 """ This function prints out all the employees in the data    """    
c.execute(""" SELECT * FROM employees """)    
return c.fetchall()conn.commit()
```

函数`get_employees()`使用方法`c.execute`中的 SQL 命令`SELECT * FROM employees`来获取所有雇员的信息。然后，该函数返回 SQL 命令`c.fetchall()`，该命令打印出方法`c.execute`中提到的所有内容。

## **创建 employee_info()函数:**

```
def employee_info(): while True:speak_text(‘please enter the first name of the employee’) print(‘\nPlease enter the first name of the employee:-’) 
first_name = input() if re.findall(‘[a-z]’, first_name): 
  break while True: 
  speak_text(‘please enter the last name of the employee’)               last_name = input() print(‘Please enter the last name of the employee:-’)if re.findall(‘[a-z]’, last_name): 
  break while True: 
 speak_text(‘please enter the pay of the employee’)  print(‘Please enter the pay of the employee:-’)  pay_emp = input()

if re.findall(“\d”, pay_emp):  
  break user_emp = Employees(first_name, last_name, pay_emp)    

  speak_text(‘employee was successfully added to the database’)      return insert_emp(user_emp)
```

`employee_info()` 函数是程序中运行的第一个函数。当该函数运行时，它要求用户输入雇员的名、姓和工资。雇员的 id 已经由表递增。`while`循环检查用户是否为特定信息输入了正确的数据类型。获得雇员的名字、姓氏和工资后，该函数将信息存储在 employees 表中。

## 创建 select_program()函数:

```
def select_program(): 
“””
 This function makes the user select from writing manual commands to edit the employee data or use the voice bot“””
print(‘Type in voice bot to use the \”voice bot\” to enter your employee data or type in \”manual\” to enter the data manually :- ‘) 
select = input()if select == ‘voice bot’ or select == ‘ voice bot’: 
 voice_commands()elif select == ‘manual’ or select == ‘ manual’: 
 manual_commands()

else: 
 select_program()
```

`select_program()`功能询问用户是想使用语音机器人输入员工信息还是手动输入员工信息。键入“手动”将运行`manual_commands()`功能，但如果用户键入`voice_commands()`，则运行`voice_commands()`功能。

## 创建 manual_commands()函数:

```
def manual_commands():
   """    
This function is executed when the user decides to edit the data manually. The if-elif-else statements are used to update, delete or get employees    """print('1\. \nWrite \"update\" to add another employee into the data')    
print('2\. Write \"delete\" to delete an employee from the data')    print('3\. Write \"get employees\" to delete an employee from the data')    
print('update, delete, or get employees....?')    
command = input()if command == 'update' or command == ' update': employee_info()
   speak_text('Here are the employees up until now')
   print('Here are the employees up until now ' + '\n' +    str(get_employees()))       
   loop_commands_function()elif text == 'delete':   

  print("Which employee do you want me to delete. Please select the id of the employee from the data below: " + '\n'
+str(get_employees()))        
 delete_employee = input()        
 delete_emp(delete_employee)        
 speak_text('employee successfully deleted')        
 speak_text('Here are the employees up until now ')   
 print('Here are the employees up until now' + '\n' +    str(get_employees()))        

  loop_commands_function()elif command == 'get employees' or command == ' get employees':          speak_text('Here are all the employees')           
  print(str(get_employees()))        
  loop_commands_function()else:        
 print('\n' + 'Please say from one of the following commands')         
 print('\n' + manual_commands())
```

当用户决定手动输入员工信息而不是使用语音机器人时，就会运行`manual_commands()`功能。这个功能非常简单。它询问用户是否想要更新(插入)、删除或获取所有雇员的信息。如果用户输入“update ”,那么`employee_info()`函数就会运行，询问雇员的名字、姓氏和工资。如果用户输入“delete ”,程序首先打印出所有雇员及其 id，以便用户删除正确的雇员。最后，如果用户键入“get employees ”,那么程序将获得所有雇员的信息。然而，如果用户输入了错误的命令，那么程序会告诉用户它接受哪些命令。在每条`if`语句的末尾，你放入`loop_commands()`函数，询问用户是否想继续这个程序。

下面是 loop_commands()函数…..

```
def loop_commands_function(): 
“”” 
This function asks the user if he/she wants to continue with the program “”” 
print(‘Do you want me to continue? Write Yes or No’) 
cont = input() 
if cont == ‘Yes’ or cont == ‘ Yes’: 
 select_program() 
elif cont == ‘No’ or cont == ‘ No’: 
 speak_text(‘thank you for your time’) 
 speak_text(‘goodbye now’) 
 exit() 
else: 
 speak_text(‘please type in yes or no’) print(‘Please type in Yes or No!’)
```

## **最后一步:创建 voice_commands()函数**

```
def voice_commands():   
 """    This function uses the voice bot to insert, delete, or get employees from the data. Similar to the manual_commands() function, but it just uses voice.    """    
speak_text('\nSay update to insert another employee into the data....')    
speak_text('Say delete to delete the employee from the data....')    speak_text('Say get employees to get the information of all employees...')     
print('\n1\. Say update to insert the employee into the data')    print('2\. Say delete to delete the employee from the data')    print('3\. Say get employees to get the information of all employees')     
text = get_audio() if text == 'update':       
 employee_info()        
 speak_text('Here are the employees up until now ')         
 print('Here are the employees up until now ' + '\n' + str(get_employees()))        
 loop_commands_function() elif text == 'delete':        
 print("Which employee do you want me to delete. Please select the id of the employee from the data below: " + '\n' +str(get_employees()))        
 delete_employee = input()        
 delete_emp(delete_employee)        
 speak_text('employee successfully deleted')        
 speak_text('Here are the employees up until now ')         
print('Here are the employees up until now' + '\n' + str(get_employees()))        
 loop_commands_function() elif text == 'get employees':        
 speak_text('\nHere are all the employees')         print(str(get_employees()))        
 loop_commands_function() else:        
 speak_text('Please say from one of the following commands')         
 voice_commands()
```

voice_commands()与 manual_commands()功能相同，但是，它要求用户使用`get_audio`功能通过麦克风输入。用户可以说“插入”来插入一个雇员，“删除”来删除一个雇员，或者“获取雇员”来获取所有雇员的信息。每当插入或删除表中的雇员时，`speak_text()` 函数都会通知用户。

# 程序中函数的顺序

通过调用函数来执行它是很重要的。因此，您需要按照提到的顺序调用以下函数来运行程序:

1.  `employee_info()`
2.  `select_program()`

# 最后的想法

如果这个程序有一些奇怪的函数名，我很抱歉，但是，我尽力解释了这个程序和它的函数。随意使用[源代码](https://github.com/hassanAli123-stack/SQLITE3-project)来为程序增加功能或者改变函数名。

与数据库交互是数据科学家最有价值的工具之一，因此，如果你像我一样正在成为一名数据科学家，这个项目可以成为你与数据库交互的一次很好的学习经历。

这个项目不仅对我来说是一个学习的机会，也是一个里程碑，因为它是我第一个致力于 GitHub 的项目。然而，任何新手数据科学家的旅程都不会就此停止…

*如果你想要更多这样的项目，请务必留下掌声和/或反馈；)*