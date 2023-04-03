# Deno- JavaScript 和 TypeScript 运行时

> 原文：<https://medium.com/analytics-vidhya/deno-javascript-and-typescript-runtime-557483bf7f22?source=collection_archive---------16----------------------->

![](img/8b6c9da63486942b7347c888de290b3e.png)

【图片来源:】[*【https://pexels.com】*](https://pexels.com/)

## 介绍

大约两年前，Node.js 的创始人 Ryan Dahl 谈到了他对 Node.js 感到遗憾的十件事。同时，他介绍了 DENO，这是一种新的、安全第一、无 npm 的 JavaScript 和 typescript 运行时的原型。最近 [DENO 1.0](https://deno.land/) 发布。

## 为什么是德诺？

我们知道 javascript 是经过战场考验的 web 动态语言，我们无法想象没有 JavaScript 的 web 行业。通过像 ECMA 国际这样的标准组织，这种语言每天都在发展。很容易解释为什么是动态语言工具的自然选择，无论是在浏览器环境中还是作为独立的进程。

NodeJS:开源、跨平台、JavaScript 运行时环境，由同一作者在大约十年前发明。人们发现它对 web 开发、工具、创建服务器和许多其他用例都很有用。在演示中，我们将详细讨论 10 件令人遗憾的事情。

现在，不断变化的 JavaScript 世界，以及像 TypeScript 这样的新补充，构建节点项目可能会成为一项有问题的工作，涉及管理构建系统和另一种剥夺动态语言脚本乐趣的重型工具。此外，链接到外部库的机制基本上是通过 NPM 库集中的，这不符合网络的理想。

## 德诺

Deno 是一个新的运行时，用于在 web 浏览器之外执行 JavaScript 和 TypeScript。Deno 试图为快速编写复杂功能的脚本提供一个完整的解决方案。[代码]

## 会取代 NodeJS 吗？

NodeJs 是一个经过战场考验的平台，并且得到了令人难以置信的良好支持，它将每天都在发展。

## 类型脚本支持

在引擎盖下，deno 基于 V8、Rust 和 Tokio 构建。`rusty_v8`箱为`V8's C++ API`提供高质量的防锈绑定。因此，很容易解释用特定的类型脚本编写意味着我们获得了类型脚本的许多好处，即使我们可能选择用普通的 JavaScript 编写代码。所以 deno 不需要 typescript 编译设置，deno 会自动完成。

## Node 与 Deno

两者都是基于 chrome V8 引擎开发的，非常适合用 JavaScript 开发服务器端。Node 用 C++写，deno 用 Rust 和 typescript 写。Node 有正式的包管理器称为 npm，而 deno 没有包管理器，代替包管理器 deno 从 URL 调用 ES 模块。Node 使用 CommonJS 语法导入包，deno 使用 es 模块。Deno 在其所有 API 和标准库中使用现代 ECMA 脚本特性，而 nodejs 使用基于回调的标准库。Deno 通过权限提供了一个安全层。Node.js 程序可以访问用户可以访问的任何内容。

## 安装 Deno

*使用自制软件(macOS):*

`brew install deno`

*使用 Powershell*

`iwr https://deno.land/x/install/install.ps1 -useb | iex`

通过`deno --version`测试您的安装，使用`deno -help`了解帮助文本，并使用`deno upgrade`升级先前安装的 deno。

```
deno 1.0.0
    A secure JavaScript and TypeScript runtime

    Docs: [https://deno.land/std/manual.md](https://deno.land/std/manual.md)
    Modules: [https://deno.land/std/](https://deno.land/std/) [https://deno.land/x/](https://deno.land/x/)
    Bugs: [https://github.com/denoland/deno/issues](https://github.com/denoland/deno/issues)

    To start the REPL:
      deno

    To execute a script:
      deno run [https://deno.land/std/examples/welcome.ts](https://deno.land/std/examples/welcome.ts)

    To evaluate code in the shell:
      deno eval "console.log(30933 + 404)"

    USAGE:
        deno [OPTIONS] [SUBCOMMAND]

    OPTIONS:
        -h, --help                     Prints help information
        -L, --log-level <log-level>    Set log level [possible values: debug, info]
        -q, --quiet                    Suppress diagnostic output
        -V, --version                  Prints version information

    SUBCOMMANDS:
        bundle         Bundle module and dependencies into single file
        cache          Cache the dependencies
        completions    Generate shell completions
        doc            Show documentation for a module
        eval           Eval script
        fmt            Format source files
        help           Prints this message or the help of the given subcommand(s)
        info           Show info about cache or info related to source file
        install        Install script as an executable
        repl           Read Eval Print Loop
        run            Run a program given a filename or url to the module
        test           Run tests
        types          Print runtime TypeScript declarations
        upgrade        Upgrade deno executable to given version

    ENVIRONMENT VARIABLES:
        DENO_DIR             Set deno's base directory (defaults to $HOME/.deno)
        DENO_INSTALL_ROOT    Set deno install's output directory
                             (defaults to $HOME/.deno/bin)
        NO_COLOR             Set to disable color
        HTTP_PROXY           Proxy address for HTTP requests
                             (module downloads, fetch)
        HTTPS_PROXY          Same but for HTTPS
```

## 你的第一个 Deno 应用

你好世界

这是一个简单的例子来教你关于 deno 的基础知识

`deno run [https://deno.land/std/examples/welcome.ts](https://deno.land/std/examples/welcome.ts)`

发出 HTTP 请求

```
const url = Deno.args[0];
const res = await fetch(url);
const body = new Uint8Array(await res.arrayBuffer());
await Deno.stdout.write(body);
```

让我们来看看这个应用程序做了什么:

*   这里，我们将第一个参数传递给应用程序，并将其存储在变量 URL 中。
*   然后，我们向指定的 URL 发出请求，等待响应，并将其存储在名为 res 的变量中。
*   然后我们将响应体解析为 ArrayBuffer，等待响应，将其转换为 Uint8Array，并存储在变量体中。
*   我们将主体变量的内容写入 stdout。

试试下面这个例子`deno run https://deno.land/std/examples/curl.ts https://example.com`你会看到一个关于网络访问的错误。那么是什么问题呢？我们知道 Deno 是一个默认安全的运行时。这意味着我们必须明确地给程序许可去做某些特权行为，比如网络访问。

再次选拔赛`deno run --allow-net=example.com https://deno.land/std/examples/curl.ts [https://example.com](https://example.com)`

简单 TCP 服务器这是一个简单服务器的例子，它接受端口 8080 上的连接，并向客户端返回它发送的任何信息。

```
const listener = Deno.listen({ port: 8080 });
console.log("listening on 0.0.0.0:8080");
for await (const conn of listener) {
  Deno.copy(conn, conn);
}
```

出于安全原因，Deno 不允许程序在没有明确许可的情况下访问网络。要允许访问网络，请使用命令行标志:

`deno run --allow-net https://deno.land/std/examples/echo_server.ts` 要测试它，请尝试使用 netcat 向它发送数据:

```
$ nc localhost 8080
   hello world
   hello world
```

与`cat.ts`示例一样，这里的`copy()`函数也不会进行不必要的内存复制。它从内核接收一个包并发送回来，没有进一步的复杂性。

## 资源

*   [德诺周刊](https://denoweekly.com/)
*   [官方文件](https://deno.land/manual)
*   [社区不和组](https://discord.gg/deno)