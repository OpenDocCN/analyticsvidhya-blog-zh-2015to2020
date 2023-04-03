# 如何用 Python 创建 Twitter 截图

> 原文：<https://medium.com/analytics-vidhya/how-to-create-twitter-screenshots-with-python-c142ef71fda7?source=collection_archive---------7----------------------->

今天，我将向您展示使用 Python 和 Pillow 库可以实现的更多内容。对于那些不熟悉的人来说，Pillow 是一个图像库，也就是说，它可以帮助你有计划地处理图像。

在本文中，我想向您展示如何自动创建 Twitter 截图，如下所示:

![](img/aa81327a6c8f123e769a4638dce88329.png)

Twitter 截图示例

如您所见，该图像包含用户照片、用户姓名和 Twitter 句柄，以及实际的 tweet。内容不居中是一个有意识的决定。

在编码之前还有最后一件事。我们将使用第三方库 [Pillow](https://pillow.readthedocs.io/en/stable/) 。因此，您首先需要用 Python 包管理器 pip 安装它。只需在命令行中输入`pip install pillow`。如果你像我一样使用 Windows，不要忘记以管理员身份运行命令行，否则你将无法安装这个库。

现在，让我们深入研究代码。我们将一步一步地解释代码的每一部分，但是在最后有一个代码要点和完整的脚本。

从导入开始，它们非常简单:来自 Pillow 的三个模块和来自内置`textwrap`模块的一个函数。

```
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
```

仅此而已。我一会儿会讲到`wrap`，但在此之前，让我展示一些有用的常量变量，我们可以定义它们来进一步提高脚本的可读性。

```
FONT_USER_INFO = ImageFont.truetype("arial.ttf", 90, encoding="utf-8")
FONT_TEXT = ImageFont.truetype("arial.ttf", 110, encoding="utf-8")
WIDTH = 2376
HEIGHT = 2024
COLOR_BG = 'white'
COLOR_NAME = 'black'
COLOR_TAG = (64, 64, 64)
COLOR_TEXT = 'black'
COORD_PHOTO = (250, 170)
COORD_NAME = (600, 185)
COORD_TAG = (600, 305)
COORD_TEXT = (250, 510)
LINE_MARGIN = 15
```

现在让我解释一下这些将会做什么。前两个变量是字体设置，即一个用于用户信息文本(用户照片旁边的两行)，另一个用于 tweet 正文。我们使用 Arial 字体，大小分别为 90 和 110，采用 UTF 8 编码。请注意，根据`ImageFont.truetype()`查找字体的方式，您选择的字体应该与脚本在同一个文件夹中，或者在您的操作系统的默认字体目录中。尽管您也可以传递一个绝对路径。剩下的变量只是设置尺寸，颜色和坐标的地方画东西。(知道(0，0)是图像的左上角)。

接下来是图像的信息、将要绘制的所有文本以及一个名为的变量，该变量将由我们将要创建的图像文件使用。

```
user_name = "José Fernando Costa"
user_tag = "@soulsinporto"
text = "Go out there and do some fun shit, not because it makes money, but because it is fun for you!"
img_name = "work_towards_better_tomorrow"
```

现在是跟踪器变量和计算的设置。我将这一部分分为三个解释，一个解释前面提到的`wrap()`函数，一个解释`x`和`y`变量，第三个解释行高计算。

`text_string_lines = wrap(text, 37)`

`wrap()`函数在这里做的是将包含我们的 tweet 的单个字符串分割成更小的字符串。我们不能真的告诉脚本绘制一个字符串，并期望它被很好地分成文本行。相反，我们让`wrap()`替我们做这件事，返回一个字符串列表。第二个参数是`37`,意思是每行不会超过 37 个字符。

```
x = COORD_TEXT[0]
y = COORD_TEXT[1]
```

这些只是追踪器变量。`x`实际上从不改变，我创建它只是为了传递`(x,y)`作为稍后的绘图坐标，但是`y`另一方面会跟踪当前垂直位置，在该位置绘制一行文本。如您所见，这两个变量分别保存水平和垂直位置，tweet 的第一行将在这两个位置绘制。

```
temp_img = Image.new('RGB', (0, 0))
temp_img_draw_interf = ImageDraw.Draw(temp_img)line_height = [
   temp_img_draw_interf.textsize(
       text_string_lines[i],
       font=FONT_TEXT
   )[1]
   for i in range(len(text_string_lines))
]
```

(很抱歉，这个示例中的代码格式不太理想)

现在进行一些简单的计算。为了提取绘制 tweet 正文每条线所需的高度，我们首先创建一个新的`Image`。从技术上讲，图像不存在，因为它的宽度和高度都是零像素，但是我们需要创建一个这种类型的对象和它的绘图接口，这样我们就可以调用`textsize()`方法来提取使用我们之前定义的字体设置绘制每行文本所需的高度。`textsize()`返回一个两项元组，即画线所需的宽度和高度，但我们只对第二个值感兴趣，即索引 1 处的值(`[1]`)。

如果您不熟悉用于`line_height`变量的语法，这是一种特殊的 Python 语法，称为 list comprehension(简称 list comp)。本质上，这是一个 for 循环，我们遍历存储在`text_string_lines`列表中的每一行文本，并对每一行调用`textsize()`方法来提取绘制该行所需的高度。因此，`line_height`是一个整数列表，表示为 tweet 正文绘制每行文本所需的高度。关于列表理解的更多内容，我强烈推荐阅读关于这个问题的 Python 文档。

现在，我们终于到了脚本的最后一部分，实际的图像创建和绘制。再说一遍，让我们一次一大块。

```
img = Image.new('RGB', (WIDTH, HEIGHT), color='white')
draw_interf = ImageDraw.Draw(img)
```

这就是你之前看到的创建那个临时的`Image`对象，但是现在要创建我们的“真实的”`Image`对象。图像使用 RGB 颜色模型，宽度为`WIDTH`像素，高度为`HEIGHT`像素，背景图像为白色(白色是模块中的命名颜色，因此我们可以使用名称而不是传递 RGB 值)。下一行为图像创建绘图接口，顾名思义，它负责图像中的任何绘图操作。

```
draw_interf.text(COORD_NAME, user_name, font=FONT_USER_INFO, fill=COLOR_NAME)
draw_interf.text(COORD_TAG, user_tag, font=FONT_USER_INFO, fill=COLOR_TAG)
```

在这几行中，我们简单地画出用户名，然后是用户句柄。传递的参数列表是:开始绘制内容的坐标、要绘制的内容(两种情况下都是文本)、字体设置和用于绘制的颜色。只是一个细节，`COLOR_NAME`是一个命名的颜色，黑色，而`COLOR_TAG`使用 RGB 值，作为三值元组传递。

```
for index, line in enumerate(text_string_lines):
    draw_interf.text((x, y), line, font=FONT_TEXT, fill=COLOR_TEXT) 
    y += line_height[index] + LINE_MARGIN
```

上面的块是我们从一开始就在谈论的那些文本行的地方。简而言之，我们遍历行列表并绘制每一行，就像我们处理用户名和句柄一样。

具体来说，我们使用 neat `enumerate()`函数，这样每次迭代我们都可以访问文本行(`line`)和它在列表中的索引(`index`)。在循环的第一行，我们绘制了`line`，与用户名和句柄完全相同，但是这里的绘制坐标是由我们之前讨论过的`x`和`y` tracker 变量给出的。在下面一行中，我们更新了`y`，它是绘制文本的垂直位置的跟踪器。我们用绘制刚刚绘制的文本行所需的高度来增加它的当前值，然后我们还增加一些距离，作为行间距。

现在，我们所缺少的是添加用户照片，这样图像就完整了。在解释如何完成最后一部分之前，让我先说一下，这张用户照片应该是一个 250x250 的圆形，四角是透明的，因为本例中使用的值和尺寸都是针对这个尺寸优化的。当然，你可以随意摆弄这些东西，创建你自己的图像。毕竟，这就是本文的目的，我只是给你(新的)工具来帮助实现你的创造力！

```
user_photo = Image.open('user_photo.png', 'r')
```

所以，首先我们加载用户照片。如果照片与脚本在同一个目录中，您可以使用示例中的文件名，但如果不是，您需要提供其路径。第二个参数`r`表示照片是以读取模式加载的，即不能修改。

```
img.paste(user_photo, COORD_PHOTO, mask=user_photo)
```

这是剧本的倒数第二行。我们将刚刚加载的照片粘贴到我们的工作图像上。第一个参数是要粘贴的图像，第二个参数是要粘贴的图像的坐标，第三个参数是要在该过程中使用的遮罩。

到目前为止，前两个论点很容易理解，但第三个可能会引起混乱。你看，当你粘贴图像时，如果没有指定蒙版，那么加载的图像将被粘贴到你的工作图像之上，忽略它的透明度。因此，如果我们也使用要粘贴的图像作为自己的蒙版，那么 Pillow 只会将图像粘贴到蒙版覆盖的区域的顶部。蒙版没有将透明胶片算作绘图区域，因此结果是我们只在工作图像中粘贴了一个 250x250 的圆形，而不是一个 250x250 的正方形。

```
img.save(f'{img_name}.png')
```

我们终于走到了尽头。图像的修改已经完成，所以我们可以把它作为新文件保存在我们的电脑上。如果您不熟悉字符串前的 f，那是 f-strings，这是 Python 3.6 中引入的一种更好的字符串格式。如果你不熟悉这个教程，我鼓励你去看看[，因为这是用 Python 格式化字符串的最好方法。换句话说，我们将图像保存为 PNG 文件(因此。png 扩展名)，其名称(字符串)设置在`img_name`变量中。](https://www.youtube.com/watch?v=nghuHvKLhJA)

仅此而已。通过这篇文章，你已经学会了如何用 Pillow 创建新的图像，如何绘制文本以及如何粘贴图像(带透明度)。正如你所看到的，这个库对于用编程的方式处理图像非常有用，而且不止于此。您可以绘制形状等，修改图像的大小，添加过滤器，等等。

我要给你的最后一样东西是完整脚本的代码要点。

现在去玩这个神奇的图书馆，并自动创建自己的图像。

像往常一样，欢迎任何和所有的反馈来改进未来的文章:)