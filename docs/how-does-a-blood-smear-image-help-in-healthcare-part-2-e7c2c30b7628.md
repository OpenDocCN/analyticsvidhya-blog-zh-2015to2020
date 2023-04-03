# 血液涂片图像对医疗保健有什么帮助？(第二部分)

> 原文：<https://medium.com/analytics-vidhya/how-does-a-blood-smear-image-help-in-healthcare-part-2-e7c2c30b7628?source=collection_archive---------22----------------------->

![](img/f52f79b466a2176ad61468308f0dbed9.png)

厚薄血涂片(P . C:[https://pix nio . com/science/biology-pictures/good-examples-of-appearance of the thick-and-Thin-blood-smears-to-be-examination-the-being-of-examination-the-under-the-microscope](https://pixnio.com/science/biology-pictures/good-examples-of-the-appearance-of-thick-and-thin-blood-smears-to-be-examined-under-the-microscope))

血液涂片检查是一种非常高效且经济的检查，通常用于以非常高的精度识别血细胞中的异常，帮助医生诊断某些血液疾病或其他医疗状况。正如我们所知，红细胞在全身携带氧气，白细胞与感染和其他炎症疾病作斗争，血小板对凝血最重要。在我之前的[博客](/analytics-vidhya/object-segmentation-using-fuzzy-divergence-in-python-a-case-study-over-peripheral-blood-smears-de61ce5dc8d1)中，我已经讨论了由于这三种类型的细胞和细胞参数之间的不平衡而可能发生的异常。在这里，我将描述血液涂片中描述一个人的健康和疾病的因素和特征。

**为什么要从图像的其余部分分割出血细胞？**

为了识别涂片特征，例如血细胞的计数、形状和总体外观，以便可以检查涂片并将其宣布为健康或异常，我们需要从涂片的其余部分中分割血细胞。

**有哪些医疗参数可用于确定特定疾病/疾病阶段的身体状况/阶段？**

就血细胞的特征而言，其他每种疾病都是不同的。血细胞大小、形状、颜色和数量的异常有助于精确检测疾病。根据这些特征，医生推断出病人的健康状况。

![](img/f8703ae30143d435e2f1d4d362772eef.png)

[https://www.google.com/imgres?imgurl = https % 3A % 2F % 2f webpath . med . Utah . edu % 2f JPEG 5% 2f heme 002 . jpg&imgrefurl = https % 3A % 2F % 2f webpath . med . Utah . edu % 2f utorial % 2f iron % 2f iron 002 . html&docid = q 5 ytitiigkajfm&TB NID = edy 2 zwjgdm 1 dxm % 3A&vet](https://www.google.com/imgres?imgurl=https%3A%2F%2Fwebpath.med.utah.edu%2Fjpeg5%2FHEME002.jpg&imgrefurl=https%3A%2F%2Fwebpath.med.utah.edu%2FTUTORIAL%2FIRON%2FIRON002.html&docid=q5YTiTIigkAJfM&tbnid=eDY2ZwJgDm1DXM%3A&vet=10ahUKEwjWmNjXn93lAhXdknAKHTGtBc4QMwhfKAMwAw..i&w=504&h=331&bih=588&biw=1315&q=normal%20blood%20smear%20images&ved=0ahUKEwjWmNjXn93lAhXdknAKHTGtBc4QMwhfKAMwAw&iact=mrc&uact=8)

*   比如**缺铁性贫血**的情况，涂片中没有观察到正常的红细胞。一个正常的红细胞是**圆形**形状和**粉红色**颜色。但是由于缺铁，我们的身体变得不能产生正常的红细胞。同样，在**镰状细胞贫血**的情况下，红细胞有一个异常的月牙形。在另一种情况下，**真性红细胞增多症**是一种红细胞疾病，身体产生过量的红细胞。
*   **血小板数量非常低**表示**血小板减少症**，是一种导致血液中血小板数量非常低的感染。
*   在**慢性髓系白血病** (CML)的病例中，白细胞计数增加，并且经常异常升高。此外，还观察到白细胞的特定模式。成比例地，原始细胞比成熟细胞和完全成熟的白细胞更高。原始细胞通常不存在于健康个体的血液中。根据 CML 期/阶段的严重程度，观察到红细胞数量减少，血小板数量增加或减少。

![](img/87a1b861fc37ceb9f5a223ec2f5e875e.png)

[https://www.google.com/imgres?img URL = http % 3A % 2F % 2fwww . pathologyoutlines . com % 2f images % 2f marrow % 2f 250 caption . jpg&imgrefurl = http % 3A % 2F % 2fwww . pathologyoutlines . com % 2f topic % 2 fmyeloproliferativecml . html&docid](https://www.google.com/imgres?imgurl=http%3A%2F%2Fwww.pathologyoutlines.com%2Fimages%2Fmarrow%2F250caption.jpg&imgrefurl=http%3A%2F%2Fwww.pathologyoutlines.com%2Ftopic%2Fmyeloproliferativecml.html&docid=tJvIly-Vpbn3IM&tbnid=V1VbXUmIs-P3YM%3A&vet=10ahUKEwjq4d2end3lAhUHuo8KHSEMAaEQMwhPKAMwAw..i&w=800&h=602&bih=637&biw=1315&q=cml%20images%20blood%20smears&ved=0ahUKEwjq4d2end3lAhUHuo8KHSEMAaEQMwhPKAMwAw&iact=mrc&uact=8)

*   血液涂片检查偶尔有助于检测导致疟疾、丝虫病和真菌感染的寄生虫。

![](img/d14a58eda253fa2634d36e9afc62a832.png)

[http://the conversation . com/how-our-red-blood-cells-keep-evolution-to-fight-malaria-96117](http://theconversation.com/how-our-red-blood-cells-keep-evolving-to-fight-malaria-96117)

当从业者可以很容易地识别人工过程时，AI/ML 驱动的自动化过程的需求是什么？

没有这样的手册涵盖了医学案例的困境。手动处理主要依赖于手动观察，其中医师需要专业知识来确认疾病状况。实际上，所有这些手动发现和在显微镜下观察到的行为现象都有助于医师和医生找到特征，检测问题的原因及其严重性。很多时候，医生发现观察结果并不全面，无法得出结论。因此，他们认为有必要进行进一步的分析/检查。显微镜涂片分析的自动化有助于减少因缺乏专业知识而导致的**观察者间差异**误差，并优化昂贵的诊断程序。在大多数情况下，第一次是正确的(RFT)。**更快的诊断**，最大限度地减少检测时间，避免人为误解，从疾病的初始阶段就为患者提供合适的治疗。还减少了检查时的重复次数，给个体带来极大的舒适。

**血液涂片图像分析面临的基本挑战是什么？**

*   血液涂片形成缺乏适当的标准化流程
*   标准化化学染色过程的变化，可能会因实验室而异。
*   人工解释会造成不同实验室的观察者之间的差异，并且在医疗从业者之间也会有所不同。
*   有时，涂片中会出现异常，如破碎细胞或污迹细胞(称为伪影)，在分析涂片时应考虑这些异常。
*   手动处理和图像捕获设备可能导致拖尾图像质量差。
*   有效的基础事实生成方法对于满足日益增长的验证和机器学习需求非常重要。现在，实践者手动地建立基础真理，其中可能存在观察者之间的差异。
*   需要用于分析异质图像数据的通用算法。

就对更准确、更可靠和更快速的算法的持续需求以及针对特定应用的专用自动化算法解决方案(使用 AI/ML)而言，这些挑战是非常互补的。

**基于图像分析能举一反三的基本参数有哪些？**

形状、大小、颜色和纹理是我们可以从血涂片的分割细胞中提取的基本参数，这些信息有助于区分正常和患病情况。

*   **形状提取**:血细胞的形状对从业者的解读影响更深；例如，在镰状细胞贫血的情况下，大量镰状细胞出现在血涂片中，其形状像镰刀或新月。

![](img/b3772caa29c3b0782320af6159bbf1a9.png)

[https://www . mayo clinic . org/-/media/kcms/GBS/patient-consumer/images/2013/08/26/10/24/ds 00324 _ im 01729 _ r7 _ sicklecellsthu _ jpg . jpg](https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/08/26/10/24/ds00324_im01729_r7_sicklecellsthu_jpg.jpg)

自动分割过程可以从图像中识别出这些月牙形，并通知疾病的严重程度，以便早期治疗可以减轻疼痛，并有助于预防与疾病相关的问题。

*   **尺寸提取**:分割出的细胞可以借助面积参数来描述。我们可以检查大小作为正常情况和患病情况的判别和重要参数。
*   **颜色提取**:颜色是另一个可以帮助识别血涂片中不同细胞的明显特征。因此，颜色是唯一有助于在涂片中识别所需对象的因素，另一方面从血液涂片图像的其余部分中丢弃其他不需要的对象。
*   **纹理提取**:纹理是识别几种基本影响红细胞的寄生虫感染的另一个维度。红细胞的结构变化有助于检测寄生虫(如疟原虫)的存在及其阶段/时期。

在我接下来的博客中，我们将看到机器学习如何借助血细胞的形状/大小/纹理信息来帮助识别疾病。在随后的博客中，我们还将探讨这种分析如何有助于疾病的早期检测和预防。如果你有任何关于血液相关疾病的问题，请发短信给我。回头见！..

链接:[https://www.linkedin.com/in/dr-madhumala-ghosh-75650b33/](https://www.linkedin.com/in/dr-madhumala-ghosh-75650b33/)