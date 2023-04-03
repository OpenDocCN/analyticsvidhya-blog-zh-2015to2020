# 移动设备上的现代 C++挑战——第一个问题

> 原文：<https://medium.com/analytics-vidhya/the-modern-c-challenge-on-mobile-the-first-problem-993a49c9cb4d?source=collection_archive---------22----------------------->

你好！我是 Xavier Jouvenot，这是关于[现代 C++挑战](https://amzn.to/2QdYmvA)的系列文章的第一部分。在这篇文章中，我将解释我如何在 C++中解决第一个问题，以及我如何在一个 Android 项目中集成该解决方案。

第一个问题的目标很简单。我们必须计算所有可被 3 或 5 整除的自然数的和，直到用户给定的某个极限，并且我们必须打印它。该解决方案将在 C++中计算，获取用户输入和显示结果的界面将由 Android Studio 框架处理。

# C++解决方案

正如你可能已经在我之前关于用 C++ 安装 [Android Studio 的博文中看到的。我们可以在 Android Studio 中使用 C++17，所以这就是我要用的。😉](https://10xlearner.com/2020/03/16/setting-up-android-studio-with-c-on-windows/)

当我开始 Android 开发之旅时，我已经将我的解决方案直接集成到 Android Studio 默认创建的文件`native-lib.cpp`中。这是这个函数的样子:

```
[[nodiscard]] constexpr auto sumOf3and5MultipleUpTo(const unsigned int limit) {
  size_t sum = 0;
  for(auto number = limit; number >= 3; --number) {
    if(number % 3 == 0 || number % 5 == 0) { sum += number; }
  }
  return sum;
}
```

让我们仔细看看这个函数，以确保我们都了解它是如何工作的，以及为什么会这样。首先，让我们看看原型和它给我们的信息:

```
[[nodiscard]] constexpr auto sumOf3and5MultipleUpTo(const unsigned int limit)
```

这个函数是 [constexpr](https://en.cppreference.com/w/cpp/language/constexpr) ，因为如果我们愿意，计算可以在编译时执行。返回类型由 [auto](https://en.cppreference.com/w/cpp/language/auto) 关键字自动推导。它也作为属性 [[[nodiscard]]](https://en.cppreference.com/w/cpp/language/attributes/nodiscard) 表示该函数的结果必须被程序使用。最后，我们有一个输入，它是一个正自然数，因为它的类型是`unsigned int`。

该函数定义的一个限制是输入不应超过类型`unsigned int`的限制`4294967295`。这个限制可以通过使用`size_t`或大整数实现来克服，但是，我现在不知道如何将这些类型链接到 Android 框架。😝

在原型之后，该函数为计算结果声明并初始化变量:

这是一个`size_t`变量，以确保我们可以存储尽可能多的数字，直到`18446744073709551615`。如果我们想得到更大的数字，我们必须使用大整数实现。

然后，我们有结果的计算:

```
for(auto number = limit; number >= 3; --number) {
  if(number % 3 == 0 || number % 5 == 0) {
    sum += number;
  }
}
```

循环从用户给定的极限下降到`3`。不需要检查到`0`，因为`1`和`2`都不能被`3`或`5`整除，即使`0`能被它们整除，把`0`加到一个和上也不会改变结果。

用`number % 3 == 0`检查数字是否能被`3`整除，用`number % 5 == 0`检查数字是否能被`5`整除。如果其中一个检查为真，我们将该数字添加到当前计算机`sum`。

最后，我们返回结果:

所以下面是我对这个问题的解决方案，我们来看看 Android Studio 中的界面是怎么做的。

# Android Studio 上的用户界面

由于这是这个应用程序目前解决的唯一问题，我们的应用程序上只有一个屏幕，所以我在 Android Studio 默认创建的文件`activity_main.xml`中直接指定了 UI 元素。

声明:由于我只是开始了我的 Android Studio 和移动开发之旅，我在这里和未来的帖子中所学到的东西对你来说可能是微不足道的。甚至可能有比我个人的解决方案更好的方法来完成同样的事情。如果你心中有任何这些方法，请在评论中告诉我，以便我可以提高自己。先谢谢你了🙂

为了创建我们的 UI，我们需要两种元素:[文本视图](https://developer.android.com/reference/android/widget/TextView)和[编辑文本](https://developer.android.com/reference/android/widget/EditText)。`EditText`将允许用户给我们输入，`TextView`将允许我们向用户显示结果。这些元素的 xml 如下所示:

```
<TextView android:id="@+id/result" android:layout_width="wrap_content" android:layout_height="wrap_content" android:text="Result:" app:layout_constraintBottom_toBottomOf="parent" app:layout_constraintLeft_toLeftOf="parent" app:layout_constraintRight_toRightOf="parent" app:layout_constraintTop_toTopOf="parent" /> <EditText android:id="@+id/plain_text_input" android:layout_height="wrap_content" android:layout_width="match_parent" android:inputType="number" android:hint="Please enter a number." app:layout_constraintBottom_toBottomOf="parent" app:layout_constraintHorizontal_bias="0.498" app:layout_constraintLeft_toLeftOf="parent" app:layout_constraintRight_toRightOf="parent" app:layout_constraintTop_toTopOf="parent" app:layout_constraintVertical_bias="0.4" android:autofillHints="" />
```

让我们更深入地挖掘这些元素。

首先是`EditText`。它有一个`android:id`属性，指定了一个 id，当我们需要在`EditText`中获得用户输入时，这个 id 会很有用。属性`android:inputType`被设置为`number`，这迫使用户将数字输入到`EditText`中。最后，我想告诉你的最后一个属性是属性`android:hint`，当`EditText`为空时，它显示一个文本来给用户一个指令或消息

现在，我们来看看`TextEdit`。像`EditText`一样，它有一个`android:id`属性，指定一个 id，当需要显示 cpp 算法找到的结果时，这个 id 会很有用。我们还有属性`android:text`，它是显示给用户的`TextView`的文本。这是我们将用 C++中的计算结果来修改的属性。目前，它只显示“结果:”。

我们不会关注其他元素，因为它们与这篇博文的目的无关，因为它们主要指定了元素的维度和位置。

# 使用 C++本机代码

为了链接和使用我们之前创建的 C++函数，我们需要做几件事情。首先，我们需要在 C++文件中声明我们希望程序的其余部分可以访问的内容。它看起来像下面的代码:

```
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_themoderncppchallenge_MainActivity_Sum3And5Multiples(JNIEnv* env, jobject /* this */, const jint i) {
  auto sum = sumOf3and5MultipleUpTo(i);
  return env->NewStringUTF(std::to_string(sum).c_str());
}
```

这个代码示例定义了一个在 Java 中使用的函数`Sum3And5Multiples`。该函数接受一个`jint` in 参数，该参数将是用户的输入，将该参数传递给 out C++函数，并获得计算结果，然后将其作为字符串返回。下面是这个函数的原型在 Java 代码中的样子:

```
public native String Sum3And5Multiples(int i);
```

现在，让我们看看在用`Edit Text`获得用户输入后，如何使用该函数在`Text View`中显示结果。因为我们的应用程序目前只解决了一个问题，所以这段代码直接写在`MainActivity.java`文件内的方法`MainActivity::onCreate`中。它看起来是这样的:

```
EditText et = findViewById(R.id.plain_text_input);
et.addTextChangedListener(new TextWatcher() {
  @Override
  public void beforeTextChanged(CharSequence s, int start, int count, int after) { } @Override
  public void onTextChanged(CharSequence s, int start, int before, int count) {
    int limit = 0;
    if(count != 0) {
      limit = Integer.parseInt(s.toString());
    }
    TextView tv = findViewById(R.id.result); tv.setText("Result: " + Sum3And5Multiples(limit));
  } @Override
  public void afterTextChanged(Editable s) { }
});
```

在这段代码中，我们通过使用它的 Id 获得了`EditText`，并在它的文本上附加了一个监听器。

```
EditText et = findViewById(R.id.plain_text_input); et.addTextChangedListener(new TextWatcher() {
```

然后，我们指定当`EditText`内的文本改变时，当用户修改输入时，我们要做什么。我们首先检查是否有输入要获取，如果没有，我们计算`0`的总和作为输入。

```
int limit = 0;
if(count != 0) {
  limit = Integer.parseInt(s.toString());
}
```

一旦我们有了用户的输入，我们就得到了`TextView`。最后，我们调用 C++方法并显示其结果。

```
TextView tv = findViewById(R.id.result);
tv.setText("Result: " + Sum3And5Multiples(limit));
```

瞧，我们已经把一切都链接到一个程序中，解决了现代 C++挑战的第一个问题[。我个人加了一些`TextView`给用户更多的信息。这是它的样子:](https://amzn.to/2QdYmvA)

# 结论

Android Studio 中的第一个问题和第一个真正的程序对我来说是一次很好的学习经历。我真的学到了很多关于 Android Studio 的知识，以及原生 C++程序的不同部分是如何工作的。

我知道我有很多东西要学，我很兴奋地学习它，就像我喜欢学习我在这篇文章中揭示的东西一样🙂请告诉我，如果你有任何建议或意见，对一些内容或对我提出的解决方案。

感谢大家阅读这篇文章，直到我的下一篇文章，有一个精彩的一天😉

# 有趣的链接

- [constexpr 文档](https://en.cppreference.com/w/cpp/language/constexpr)

- [文本视图文档](https://developer.android.com/reference/android/widget/TextView)

- [编辑文本文档](https://developer.android.com/reference/android/widget/EditText)

- [GitHub 完整解决方案](https://github.com/Xav83/TheModernCppChallenge_AndroidStudio/releases/tag/v0.0.1_FirstProblem)

*原载于 2020 年 3 月 23 日 http://10xlearner.com*[](https://10xlearner.com/2020/03/23/the-modern-c-challenge-on-mobile-the-first-problem/)**。**