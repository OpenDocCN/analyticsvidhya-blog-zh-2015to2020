# 我用 Python 做了一个非接触式秘密圣诞老人算法

> 原文：<https://medium.com/analytics-vidhya/i-made-a-contactless-secret-santa-algorithm-with-python-7374d4a79c56?source=collection_archive---------9----------------------->

![](img/9ce140830a3cf69bd77e156f292b6acd.png)

罗曼·桑博尔斯基在 Dreamstime.com 拍摄的照片

今年，我的朋友们给了我一个任务:组织今年的神秘圣诞老人。通常这不是我的角色。但是通常组织它的人已经离开我们去英国喝茶和会见女王了。现在，你可能会问，为什么不把名字写在纸上，揉成一团，塞进帽子里呢？你为什么不像正常人那样做呢？然而，你可能注意到了，也可能没有注意到，我们现在正处于疫情。没有正常的做事方式，除非你指的是新常态。所以我需要让它无接触，便于我携带——因为我很懒。

我们可以把名字贴在帽子里，然后一个一个地挑选。然而，a)我们组有 5 个人以上，这是非法的，b)这需要一段时间，如果我选了自己的名字，我们必须重新开始，我可能会痛苦地哭。我可以只分配名字，但这意味着我被排除在外。没有给我的礼物，没有用我的超级侦探能力找出谁得到了谁。我做不到。不，我知道我必须用 Python 创建一个算法！

所以我走到绘图板前，列出了这个程序的功能:

*   获取所有参与者的姓名和电子邮件并存储起来。
*   给每个参与者分配一个神秘的圣诞老人，确保每个人都有一个不属于自己的名字。
*   通过电子邮件向每个人发送他们的分配，实现真正的无接触体验！

首先，我需要找到工作所需的 Python 库。有不少:

```
**import** re **import** random
**import** smtplib
**from** email.mime.multipart **import** MIMEMultipart
**from** email.mime.text **import** MIMEText
```

随机库是这样的，我可以使用 **randint** 函数来随机化圣诞老人的秘密分配。其余的库是我的程序的电子邮件部分所必需的。re 库是这样我可以验证电子邮件！

我还初始化了几个变量:

```
sender_address = 'email'
sender_pass = 'password'names = []
emails = []
recipient = []
budget = 50count = 0
```

接下来，我需要决定如何将名字和电子邮件放入程序中。我决定有两种数据输入方法:

*   **通过文字(。txt)文件**:用户设置一个特定格式的文本文件*名称，email* 。每一个新的行都有一个新的参与者，程序将处理文本文件并将所有的参与者存储在列表中。
*   **手动输入**:用户手动输入每个参与者的姓名和邮箱，程序会提示(“输入参与者 1 的姓名:”、“输入参与者 1 的邮箱:”)等。).

下面是数据输入代码的样子(包括验证！):

```
**# asks the for data entry method**
print("""Welcome to the secret santa decision-maker! How would you like to enter the information?
   1\. give a text (.txt) file with format of: name, email address
   2\. manually enter information""")x = 0
while(x == 0):
   try:
      option = int(input("Info entry method (1 or 2): "))
      if(option > 2 or option < 1):
         print("ERROR: You can only input 1 or 2!")
         print("""How would you like to enter the information?
            1\. give a text (.txt) file
            2\. manually enter information""") else:
      x = 1
   except ValueError:
      print("ERROR: Please input 1 or 2!")
      print("""How would you like to enter the information?
         1\. give a text (.txt) file
         2\. manually enter information""")**# gets the number of participants**
x = 0
while(x == 0):
   try:
      count = int(input("Enter number of participants: "))
      if(count < 2):
         print("ERROR: Number of participants must be 2 or more!")
      else:
         x = 1
   except ValueError:
      print("ERROR: Please input a valid integer number!")**# option 1: read the file (assumes file format is correct)**
if(option == 1): 
   x = 0
   while(x == 0):
      filename = str(input("Name of text file (must end in .txt): "))
      if (filename[-4:] == '.txt'):
         x = 1 
      else:
         print("ERROR: Please input a file name which ends with .txt")
   text = open(filename, "r") 
   for i in range(0, count):
      info = text.readline().split(', ')
      names.append(info[0])
      emails.append(info[1])**# option 2: manually get info (assumes email is correct)**
elif(option == 2):

   # for validating an Email
   regex **=** '^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$' print("Ok! It's time for you to input the participants' information!")
   for i in range(1, count + 1):
      name = str(input(f'Enter participant the name of participant {i}: '))
      names.append(name)
      x = 0
      while(x == 0):
         email = str(input(f'Enter the email of participant {i}: '))
         if(re.search(regex, email)):
            emails.append(email)
            x = 1
         else:
            print("ERROR: invalid email!")
```

现在是节目的主要部分:神秘的圣诞老人分配。首先，我复制了 **names** 列表——我将从这个列表中选择分配，并在分配后删除该名称。然后，我使用了一个 **for 循环**来遍历**名字**列表中的每个名字，并分配他们将成为谁的秘密圣诞老人。为此，我将使用内置 Python **random** 库中的 **randint** 函数。这个随机数代表收件人的**索引**。然后，我将收件人的名字添加到**收件人**列表中。然而，有一些情况我必须检查:

*   如果收件人的名字和神秘圣诞老人的名字一样。这是不可能的，因为你不能成为自己的秘密圣诞老人！因此，我设置了一个 **if，else** 自变量。如果名称相同，程序将再次运行 **randint** 函数来选择另一个索引。
*   如果唯一留下的收件人名字和神秘圣诞老人的名字一样。如果我继续使用 randint 函数，我将陷入无限循环，因为没有其他选择！因此，我有另一个 **if** 语句，检查 **possible_santa** 列表的长度是否为 1(意味着程序必须再次运行)。这将**重做**布尔设置为**真**，允许程序再次运行。

```
possible_santa = names.copy()cont = 0while(cont == 0):

    redo = False

    possible_santa = names.copy()

    for i in range(0, len(names)):
        recip = random.randint(0, len(possible_santa) - 1)
        x = 0
        while(x == 0):
            if(names[i] == possible_santa[recip]):
                if(len(possible_santa) == 1):
                    redo = True
                    x = 1
                else:
                    recip = random.randint(0, len(possible_santa) - 1)
            else:
                x = 1
        if(redo != True):
            recipient.append(possible_santa[recip])
            possible_santa.pop(recip)
            cont = 1
        else:
            cont = 0
```

既然分配已经设置好了，我需要将它们通过电子邮件发送给参与者！为此，我使用了 SMTP——Python 有一个内置的电子邮件发送库，smtplib。我还使用了 email.mime.multipart 和 email.mime.text —更多的 Python 库。我用了这个网站:[https://www . freecodecamp . org/news/send-emails-using-code-4 fcea 9 df 63 f/](https://www.freecodecamp.org/news/send-emails-using-code-4fcea9df63f/)做参考，确实很有帮助！

下面是代码片段，还有一些注释，我希望这些注释能解释一切！

```
# this code must run for each name
for i in range(0, count): # the message which will be sent in the email
    mail_content = f'''Hello {names[i]},

You are the secret santa of {recipient[i]}!

Remember the budget is ${budget}
    '''

    # sets the email address the email will be sent to
    receiver_address = emails[i]

    # sets up the MIME
    message = MIMEMultipart()
    message['From'] = sender_address # your email address
    message['To'] = receiver_address # Secret Santa's email address
    message['Subject'] = 'Secret Santa' # subject of the 

    # sets the body of the mail
    message.attach(MIMEText(mail_content, 'plain'))

    # creates the SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)
    session.connect("smtp.gmail.com", 587)
    session.ehlo()
    session.starttls()
    session.login(sender_address, sender_pass)
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
```

我还想创建一个文本文件，保存分配，以防出错！这确保了我可以仔细检查每件事，我们有一个备份列表。

为此，我使用 open 函数创建了一个新的文本文件。然后我使用了一个 for 循环来遍历每个秘密圣诞老人和接收者，并将每个分配写入文件中的一个新行！请确保您也关闭了文本文件！

```
allocations = open("SantaAllocations.txt", "w+")for i in range(0, len(names)):
    allocations.write(f'{names[i]} is the secret santa of {recipient[i]}\n')

allocations.close()
```

这不是创建这个生成器的唯一方法，如果你能想到任何其他方法，请告诉我！

我迫不及待地想使用这个程序，并得到所有的礼物，我希望你喜欢编码和使用它！祝你圣诞快乐，如果你喜欢这个，可以看看我的博客[https://itsliterallymonique.wordpress.com/](https://itsliterallymonique.wordpress.com/)了解更多类似的内容！

如果你想下载这段代码，就在这个 github 仓库里:【https://github.com/moniquethemuffin/Secret-Santa-Generator