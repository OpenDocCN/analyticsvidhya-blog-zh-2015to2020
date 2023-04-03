# 使用 Python 的聊天室

> 原文：<https://medium.com/analytics-vidhya/chat-room-server-using-python-ab34d8cb1567?source=collection_archive---------16----------------------->

T 他的文章简要介绍了使用 Python 编程语言建立聊天室并允许多个用户相互交流的想法。

![](img/dcfcf34a682f0b2908ddd0482af5af78.png)

由[在](https://unsplash.com/@hiteshchoudhary?utm_source=medium&utm_medium=referral) [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

> 在深入之前，我们先了解一下什么是插座。

# 什么是插座？

一个**套接字**是在网络上运行的两个程序之间的双向通信链路的一个端点。“当我们谈到 Python 时，套接字编程是一种连接网络上两个节点以便相互通信的方法。在 Python 代码中，我们将使用套接字库，因为它使用户能够通过网络传输信息

在本文中，我们将在网络的两端设置一个套接字，并允许客户端通过服务器与其他客户端进行交互。会有两种不同的剧本-

1.  **服务器端脚本**
2.  **客户端脚本**

我们可以在操作系统命令提示符或终端中运行这些脚本。当服务器端脚本运行时，它等待客户端连接。一旦建立了连接，客户端和服务器就可以相互通信了。

> “Python 从一开始就是 Google 的重要组成部分，并且随着系统的成长和发展而保持不变。如今，数十名谷歌工程师使用 Python，我们正在寻找更多掌握这种语言的人。”
> 
> ~ Peter Norvig，[谷歌公司的搜索质量总监](http://google.com/)

# 服务器端**脚本**

![](img/b7cc07d4bbe38d1defa02afb9244b786.png)

托马斯·詹森在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄的照片

这将是必须运行的第一个脚本，以便建立一个套接字，按照代码中的指定将主机和端口绑定在一起。现在，我们将使用 Localhost 进行演示。

**导入库:**

套接字库使用户能够通过网络传输信息。它提供了 bind()、send()、recv()、close()等各种函数。为此我们使用 TCP 套接字，因此我们使用`AF_INET`和`SOCK_STREAM`标志。导入后，让我们设置一些*常量*以备后用:

```
import socket
import select 
HEADER_LENGTH = 10 
IP = "127.0.0.1"
PORT = 4321
```

**插座连接:**

为了创建一个套接字并将其与 IP 和端口连接，我们将使用以下语句-

```
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
```

**Bind 和 Listen :** 接下来，我们将使用 Bind()和 Listen()。bind()方法将服务器绑定到特定的 IP 和端口。listen()方法将服务器置于监听模式。这允许服务器侦听新的连接。

```
server_socket.bind((IP, PORT))
server_socket.listen()
```

**创建列表并调试:**

我们现在将创建一个套接字列表来选择和跟踪我们的客户端。该列表将连接到客户端。在这里，插座被用作钥匙。

```
sockets_list = [server_socket]
clients = {}
print(f 'Listening for connections on {IP}:{PORT}...')
```

**接收消息:**

现在，服务器必须接收消息并将它们显示给连接的客户机。它执行以下步骤

*   阅读标题。
*   如果客户端以任何方式失去连接，那么 socket.close()方法将有助于返回 false 值。
*   将标头转换为长度(整数值)。
*   返回一些有意义的数据。
*   如果上述条件为假，则返回假。

```
def receive_message(client_socket):
try:
  message_header = client_socket.recv(HEADER_LENGTH)

  if not len(message_header):
      return False

  message_length = int(message_header.decode('utf-8').strip())

  return {'header': message_header, 'data': client_socket.recv(message_length)}except:
   return False
```

**阅读和传递信息:**

现在，我们必须从所有的客户端套接字读取消息，并将它们发送到各自的客户端。这是通过以下方式实现的:

*   我们将使用“while 循环”,然后我们将使用 select.select
*   我们现在将遍历 read_sockets 列表。这些是必须读取数据的套接字。
*   如果通知的套接字是我们的服务器套接字，那么这意味着我们得到了一个新的连接，我们想要处理它。
*   现在，我们想把这个新的 client_socket 添加到 sockets_list 中。
*   最后，我们将保存这个客户机的用户名，我们将把它保存为 socket 对象的键值。
*   如果客户端断开连接，那么消息将是空的。
*   假设这不是一个断开。

```
while True:
read_sockets, _, exception_sockets = select.select(sockets_list, [], sockets_list)

for notified_socket in read_sockets:

if notified_socket == server_socket:

client_socket, client_address = server_socket.accept()

user = receive_message(client_socket)

if user is False:
   continue

sockets_list.append(client_socket)

clients[client_socket] = userprint('Accepted new connection from {}:{}, username{}'.format(*client_address, user['data'].decode('utf-8')))

else:
   message = receive_message(notified_socket)if message is False:
   print('Closed connection from:{}'.format(clients[notified_socket]['data'].decode('utf-8')))sockets_list.remove(notified_socket)del clients[notified_socket]continueuser = clients[notified_socket]print(f'Received message from {user["data"].decode("utf-8")}:{message["data"].decode("utf-8")}')for client_socket in clients:

if client_socket != notified_socket:client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])for notified_socket in exception_sockets:sockets_list.remove(notified_socket)del clients[notified_socket]
```

> 不要忘记将这个文件保存为- Server.py

# 客户端

![](img/ac3da8da2cc045b0e50d04e3b2d798ea.png)

照片由[JESHOOTS.COM](https://unsplash.com/@jeshoots?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

现在我们需要一些可以与我们的服务器交互的东西。初始步骤与 Server.py 相同

> 该文件将保存为 Client.py

**导入库:**

套接字库使用户能够通过网络传输信息。它提供了 bind()、send()、recv()、close()等各种函数。为此我们使用 TCP 套接字，因此我们使用`AF_INET`和`SOCK_STREAM`标志。导入后，让我们设置一些*常量*以备后用:

```
import socket
import select
import errnoHEADER_LENGTH = 10
IP = "127.0.0.1"
PORT = 4321
my_username = input("Username: ")client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)client_socket.connect((IP, PORT))
```

**连接服务器:**

我们需要将 **s** et 连接到非阻塞状态，所以 recv()调用不会阻塞，只是返回一些异常。我们将把 recv 方法设置为不阻塞。我们的服务器希望第一条消息是用户名

```
client_socket.setblocking(False)username = my_username.encode('utf-8')username_header =f"{len(username):<{HEADER_LENGTH}}".encode('utf-8')client_socket.send(username_header + username)
```

**通过消息交流:**

我们现在已经为客户端的主循环做好了准备，它将接受来自客户端的新消息。我们将用户名附加到客户端发送的消息中。

我们也想要接收信息。为了显示消息，我们需要用户名和消息，它们都有单独的头和内容。我们会得到真实的用户名。用同样的逻辑，我们可以得到信息。然后我们将用户名和消息输出到屏幕上。

```
while True:
    message = input(f'{my_username} > ')if message:
        message = message.encode('utf-8')
        message_header = f"{len(message):<{HEADER_LENGTH}}".encode('utf-8')
        client_socket.send(message_header + message)try:
        while True:
            username_header = client_socket.recv(HEADER_LENGTH)if not len(username_header):
                print('Connection closed by the server')
                exit()
username_length = int(username_header.decode('utf-8').strip())username = client_socket.recv(username_length).decode('utf-8')message_header = client_socket.recv(HEADER_LENGTH)message_length = int(message_header.decode('utf-8').strip())message = client_socket.recv(message_length).decode('utf-8')print(f'{username} > {message}')except IOError as e:
   if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
       print('Reading error: {}'.format(str(e)))
       exit()continueexcept Exception as e:
exit()
```

> *祝贺你能坚持到现在。你现在可以建立自己的聊天室了。*你可以在这里查阅服务器端和客户端的代码[和详细注释。](https://github.com/wbhoomika/Chat_Room-using-Python)
> 
> **如果你有任何想法、评论、问题或者你只是想了解我的最新内容，请随时与我联系**[***Linkedin***](https://www.linkedin.com/in/bhoomikawavhal/)***或***[***Github***](https://github.com/wbhoomika)***。***