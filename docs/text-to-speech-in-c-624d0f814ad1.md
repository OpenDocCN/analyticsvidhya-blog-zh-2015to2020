# C++中的文本到语音转换

> 原文：<https://medium.com/analytics-vidhya/text-to-speech-in-c-624d0f814ad1?source=collection_archive---------0----------------------->

![](img/9813b5491cfbfd7957b272facd5a4a25.png)

丹尼尔·山特维克在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

## 用 C++实现文本到语音转换的简单程序

文本到语音是机器学习的一种常见实现方式，事实上，已经建立了许多使用文本到语音的伟大的机器学习应用程序。通过导入一些预定义的模型并使用它们，在 C++中进行文本到语音的转换要容易得多。让我们来看看吧。

```
import pyttsx3
engine = pyttsx3.init() # object creation""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate', 125)     # setting up new voice rate """VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
print (volume)                          #printing current volume level
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for femaleengine.say("Hello World!")
engine.say('My current speaking rate is ' + str(rate))
engine.runAndWait()
engine.stop()
```

你看，Python 和基本 TTS 系统 20 行代码。现在，让我们看看如何在 C++中实现它。

## 先决条件

*   微软 Visual Studio 带 C++或者下载了[SAPI Linux 版](https://github.com/mjakal/sapi5_on_linux)
*   熟悉指针
*   熟悉多线程和 COM 编程会有所帮助，但不是必需的

## 我们开始吧

我们将从基本的编码开始，然后我们将采用面向对象的方法。

让我们包括标题。

```
#include <sapi.h>
#include<iostream>
#include <string>
using namespace std;
```

我们将声明一个 ISpVoice 类型的指针。ISpVoice 接口使应用程序能够执行文本合成操作。

```
ISpVoice* pVoice=NULL;
```

现在我们声明一个 HRESULT 类型的变量。HRESULT 是 Windows 操作系统和早期的 IBM/Microsoft OS/2 操作系统中使用的一种数据类型，用于表示错误条件和警告条件。HRESULT 最初的目的是为公共和微软内部使用正式地安排错误代码的范围，以防止 OS/2 操作系统的不同子系统中的错误代码之间的冲突。HRESULTs 是数字错误代码。HRESULT 中的各个位对有关错误代码的性质及其来源的信息进行编码。HRESULT 错误代码是 COM 编程中最常见的，它们构成了标准化 COM 错误处理约定的基础。

```
HRESULT hr;
```

现在是一个 [wstring](https://stackoverflow.com/questions/402283/stdwstring-vs-stdstring) 类型的变量，我们将从用户那里获取输入。

```
wstring input;
```

SAPI 是一个基于 COM 的应用程序，COM 必须在使用前和 SAPI 活动期间进行初始化。在大多数情况下，这适用于宿主应用程序的整个生命周期。

```
a=CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
if (FAILED(a))
{   
cout << "ERROR 404 FAILED INITIALIZING COM\n";
exit(1);
}
```

先来了解一下 CoInitializeEx 的参数。

```
HRESULT CoInitializeEx(LPVOID pvReserved, DWORD dwCoInit);
```

第一个参数是保留的，必须为空。第二个参数指定程序将使用的线程模型。

一旦我们在工作状态中有了 COM，我们的下一件事就是创建简单的 COM 对象的声音。默认语音来自“控制面板”中的“语音”部分，包括语音(如果您的系统上有多个语音)和语言(英语、日语等)等信息。).

```
hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void **)&pVoice);
```

我们来理解一下是[参数](https://docs.microsoft.com/en-us/windows/win32/api/combaseapi/nf-combaseapi-cocreateinstance)。

`rclsid`

与将用于创建对象的数据和代码关联的 CLSID。

`pUnkOuter`

If **NULL** 表示对象不是作为聚合的一部分创建的。如果非**空**，则指针指向聚合对象的 [IUnknown](https://docs.microsoft.com/windows/desktop/api/unknwn/nn-unknwn-iunknown) 接口(控制 **IUnknown** )。

`dwClsContext`

管理新创建对象的代码将在其中运行的上下文。这些值取自枚举 [CLSCTX](https://docs.microsoft.com/windows/desktop/api/wtypesbase/ne-wtypesbase-clsctx) 。

`riid`

对用于与对象通信的接口标识符的引用。

`ppv`

接收 *riid* 中请求的接口指针的指针变量的地址。成功返回后，* *ppv* 包含请求的接口指针。失败后，* *ppv* 包含**空值**。

现在我们必须做我们的实际任务，也就是说。说话是一个简单的单行任务，我们必须调用 speak out of voice 对象。

```
if( SUCCEEDED( hr ) )
{   getline(wcin,input);    
    **hr = pVoice->Speak(**input.c_str()**, 0, NULL);
**    pVoice->Release();
    pVoice = NULL;
}
```

点击阅读更多关于 Speak [参数的信息](https://docs.microsoft.com/en-us/previous-versions/windows/desktop/ms719820(v%3Dvs.85))

结合我们得到的一切

```
#include <sapi.h>
#include<iostream>
#include <string>
using namespace std;int main()
{
  ISpVoice* pVoice=NULL;
  HRESULT hr;
  wstring input;
 a=CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
 if (FAILED(a))
 {   
 cout << "ERROR 404 FAILED INITIALIZING COM\n";
 exit(1);
 }
 HRESULT CoInitializeEx(LPVOID pvReserved, DWORD dwCoInit);
 hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL,       IID_ISpVoice, (void **)&pVoice);

if( SUCCEEDED( hr ) )
{   getline(wcin,input);    
    **hr = pVoice->Speak(**input.c_str()**, 0, NULL);
**    pVoice->Release();
    pVoice = NULL;
}
return 0;
}
```

让我们用面向对象的方法来做吧

我们的文件结构如下:

**头文件**

*   基本语音. h
*   女声 h
*   男声

**源文件**

*   基础语音
*   女性之声
*   TTS.cpp
*   男声. cpp

Basic Voice.h 将成为我们的基类。让我们看看它

```
#pragma once#include <sapi.h>#include<iostream>#include <string>using namespace std;class BasicVoice{protected:int choice;ISpVoice* pVoice;HRESULT hr,a;wstring input;public:BasicVoice() {pVoice = NULL;input = L"";a=CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);//HRESULT CoInitializeEx(LPVOID pvReserved, DWORD dwCoInit);if (FAILED(a)){cout << "ERROR 404 FAILED INITIALIZING COM\n";exit(1);}hr = CoCreateInstance(CLSID_SpVoice, NULL, CLSCTX_ALL, IID_ISpVoice, (void **)&pVoice);}virtual void setSpeech();virtual void byeSpeech() = 0;virtual void outSpeech();virtual ~BasicVoice() {::CoUninitialize();delete pVoice;}};
```

在我们的基类中，我们初始化所有的变量。让我们在 BasicVoice.cpp 中查找

```
#include "BasicVoice.h"//pure abstract class so empty functionsvoid BasicVoice::setSpeech(){}void BasicVoice::byeSpeech(){}void BasicVoice::outSpeech(){}
```

**MaleVoice.h**

```
#pragma once#include "BasicVoice.h"class MaleVoice :public BasicVoice{public:void setSpeech();void outSpeech();void byeSpeech();};
```

**Malevoice.cpp**

```
#include "MaleVoice.h"void MaleVoice::setSpeech(){if (SUCCEEDED(hr)){cout << "Enter text:\n";cin.ignore(1, '\n');getline(wcin, input);}else{cout << "NOt Initalized";exit(-1);}system("cls");cout << "At What Speed you want to Play your Voice\n1 for Normal \n2 for -2x\n3 for 2x";cin >> choice;if (choice == 2)hr = pVoice->Speak((L"<rate absspeed='-2'>" + input).c_str(), 0, NULL);else if (choice == 3)hr = pVoice->Speak((L"<rate absspeed='2'>" + input).c_str(), 0, NULL);elsehr = pVoice->Speak(input.c_str(), 0, NULL);}void MaleVoice::outSpeech(){pVoice->Release();pVoice = NULL;::CoUninitialize();}void MaleVoice::byeSpeech(){}
```

请注意，我们正在设置速度。可以通过 XML 标签添加许多功能。在这里看完整个。

**女性声音. h**

```
#pragma once#include "BasicVoice.h"class FemaleVoice : public BasicVoice{public:void setSpeech();void outSpeech();void byeSpeech();};
```

**女性声音. cpp**

```
#include "FemaleVoice.h"void FemaleVoice::setSpeech(){if (SUCCEEDED(hr)){cout << "Enter text:\n";cin.ignore(1,'\n');getline(wcin, input);}cout << "At What Speed you want to Play your Voice\n1 for Normal \n2 for -2x\n3 for 2x";cin >> choice;if (choice == 2)hr = pVoice->Speak((L"<rate absspeed='-2'><voice required='Gender = Female;'>" + input).c_str(), 0, NULL);else if (choice == 3)hr = pVoice->Speak((L"<rate absspeed='2'><voice required='Gender = Female;'>" + input).c_str(), 0, NULL);elsehr = pVoice->Speak((L"<voice required='Gender = Female;'>" + input).c_str(), 0, NULL);}void FemaleVoice::outSpeech(){pVoice->Release();pVoice = NULL;::CoUninitialize();}void FemaleVoice::byeSpeech(){if (SUCCEEDED(hr)){hr = pVoice->Speak(L"<voice required='Gender = Female'> < rate absspeed = '-5' > Bhut Shukria Sir", 0, NULL);}}
```

在这里，我们通过使用 XML 标签添加了女声选项。

**TTS.cpp**

```
#include "BasicVoice.h"#include "MaleVoice.h"#include "FemaleVoice.h"int main(){BasicVoice* b1 = NULL;b1 = new MaleVoice;int choice;do {cout << "1 to Output in Male Voice \n2 to Output in Female Voice\n";cin >> choice;switch (choice){case 1:b1 = new MaleVoice; //  we create a new malevoice object.b1->setSpeech();b1->outSpeech();delete b1; //after outputing that voice , we delete that objectbreak;case 2:b1 = new FemaleVoice;// we create a new femalevoiceb1->setSpeech();b1->outSpeech();delete b1;//after outputing that voice , we delete that objectbreak;case 3:b1 = new FemaleVoice;b1->byeSpeech();b1->outSpeech();delete b1;break;default:break;}} while (choice != 3);system("pause");return 0;}
```

在这里，我们使用菜单驱动的方法来获取用户的输入，并以语音格式生成它们。

在这里寻找完整的组织代码[。](https://github.com/ahmadmustafaanis/Text-to-Speech-in-Cpp)

如果你在 C++中成功实现了文本到语音转换，请在评论中告诉我。

在[推特](https://twitter.com/AhmadMustafaAn1)上与我联系。

请在评论中留下您的反馈。