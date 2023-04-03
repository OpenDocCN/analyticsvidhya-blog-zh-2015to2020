# 微调 GPT-2 对拜登和川普的政治演讲

> 原文：<https://medium.com/analytics-vidhya/fine-tuning-gpt-2-on-biden-and-trumps-political-speeches-3d96d82f4f05?source=collection_archive---------18----------------------->

![](img/bc3a0ad6517280d906c78eb51c6c818f.png)

来自[突发](https://burst.shopify.com/?utm_campaign=photo_credit&amp;utm_content=Browse+Free+HD+Images+of+White+Feather+Quill+Laid+Next+To+An+Ink+Bottle&amp;utm_medium=referral&amp;utm_source=credit)的 [Samantha Hurley](https://burst.shopify.com/@lightleaksin?utm_campaign=photo_credit&amp;utm_content=Browse+Free+HD+Images+of+White+Feather+Quill+Laid+Next+To+An+Ink+Bottle&amp;utm_medium=referral&amp;utm_source=credit) 的原始照片

# 介绍

鉴于一天后的选举结果，我们对 GPT-2 进行了微调，以生成两位候选人拜登和川普的政治演讲，从而更好地了解他们的立场。

GPT-2 代表生成式预训练转换器 2，它用于基于自然语言处理生成新内容。

最初，GPT-2 是在 reddit 帖子上训练的，具有积极的因果关系，所以我们采用了 [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) 并重新训练了“中型”355M 超参数，用于新的政治演讲。

[在 github 查看代码。](https://github.com/Bayaniblues/political-gpt2)

# 观众

这篇文章主要是技术性的，吸引了希望学习如何微调 GPT-2 的数据科学家。任何非技术读者仍然可以阅读我们的结果，以获得对两位候选人立场的新见解。

# 培训用数据

为了处理训练数据，我们使用了 Kaggle 数据集[乔·拜登本周的演讲](https://www.kaggle.com/vyombhatia/joe-bidens-speeches-of-this-week)，以及[唐纳德·特朗普的集会](https://www.kaggle.com/christianlillelund/donald-trumps-rallies?select=LasVegasFeb21_2020.txt)。这为我们提供了大量的数据来开始训练我们的微调模型。

这是拜登最近一次演讲的摘录

```
We have a great purpose as a nation to open the doors of opportunity to all Americans, to save our democracy, to be a light to the world once again, and finally, to live up to and make real the words written in the sacred documents that founded this nation, that all men and women are created equal, endowed by their creator with certain inalienable rights among them, life, liberty, and the pursuit of happiness.
```

以下是特朗普最新演讲的摘录

```
Oh, thank you very much, everybody. Thank you. Well, thank you very much. Thank you very much. And considering that we caught President Obama and sleepy Joe Biden, spying on our campaign, treason, will probably be entitled to another four more years after that. I wanted to thank you, and Art and Brandon, you're incredible. I've known you now a long time. Right from the beginning, we had the chemistry, we had that good chemistry… Oh, sit down. It's 122 degrees in this place.
```

# 数据清理

为了训练有素的 gpt-2 模型，我们对我们的文档进行了格式化，这样 gpt-2 就可以用同样的格式来描述我们的候选人的谈话方式。

这些演讲是文本文件的形式，所以我们将这些文本文件合并到一个合并文件夹中，以便对 gpt-2 进行微调。

我们将不得不用**<| startoftext |>**和 **< |endoftext| >** 标签来分隔这些发言。这些标签很重要，因为它们是 gpt-2 分隔内容的方式。

```
import os# This creates our new txt file
outf = open("candidate.txt", 'w')# A file walk is useful to read all files in a fold
for root, dirs, files in os.walk("/content/candidate_name/", topdown=False): for name in files:
        # .write() allows us to write to the txt file f = open(os.path.join(root, name), 'r')
        outf.write('<|startoftext|>')
        outf.write("\n")
        outf.write(f.read())
        outf.write("\n")
        outf.write('<|endoftext|>')
        outf.write("\n")outf.close()
```

# 培养

我们运行我们的模型并改变超参数，以便它接受 355M 模型。我们训练了 GPT 2 号

```
sess = gpt2.start_tf_sess()gpt2.finetune(sess, dataset=file_name, model_name='355M', steps=1000, restore_from='fresh', run_name='candidate', print_every=10, sample_every=100, save_every=200)
```

# 结果

我们最终可以通过 generate 方法使用我们的模型。Gpt-2 可以保留多达 500 个令牌的内容，这些结果读起来总是很有趣。

# **拜登发表演讲**

> 我们的想法是，让哈佛商学院更多地参与我们正在进行的全国对话。举个例子，我开玩笑说……我的竞选活动的共同发起人中有两个人，非洲裔美国人，他们说自己是摩根人。我是摩根人。嗯，有很多很棒的大学，包括这里，但是我们有机会提供，你可以去那些大学。你怎么做到的？嗯，我会确保任何家庭，任何来自年收入低于 125，000 美元的家庭的人都能获得免费的大学教育。他们进不去。如果他们被录取并获得资格，他们不用支付任何费用就能上大学。
> <| endoftext |>
> <| startoftext |>
> 看，另一件事是，如果你想一想，FEMA，联邦紧急情况…他们同意为学校提供口罩。他们开始向学校分发。你猜怎么着？总统不喜欢这样，或者有人不喜欢这样。裁决是，“安全开放学校不是国家紧急状态”，所以他们停止了它。甚至拒绝为学校提供口罩，因为这不是国家紧急情况。如果这不是国家紧急事件，我不知道这到底是什么。我们做这件事的方式是…我将以此结束。我不需要告诉埃里克以及加利福尼亚州、俄勒冈州和华盛顿州的市长们。这些危机都不应该有党派因素。我是认真的。一个都没有。如果我当选总统，我不会成为民主党总统，我会成为美国总统。无论你投票支持我还是反对我，无论你的城市是红色还是蓝色，我都会在那里。我向你保证。坦率地说，我们应该能够像市长们解决问题一样解决每一个问题，通过以事实和现实为导向，把我们选民的福祉放在第一位，把人们团结在对每个人都有效的解决方案周围，但不幸的是，这不是本届政府的行事方式。
> <| endoftext |>
> <| startoftext |>
> 所以你回家你妈走了，你爸也走了。他实际上是个无名小卒？这个想法，我们谈论人的方式，这些都是我们力所能及的。病毒不是他的错，但他处理病毒的方式已经接近犯罪了。很多人都断了他们的脖子，顺便说一下，我已故的妻子，当我第一次结婚时，我娶了一个来自纽约锡拉丘兹的漂亮女人。我当选了，当时我 29 岁。我来自一个非常谦虚的家庭。不是吹牛，但我被列为国会 36 年来最穷的人。因为我认为你在那里除了工资之外不应该赚任何钱。我真的很擅长这个。这让我发疯。现在，我们的情况是，如果你看一下，没有关于如何开办学校，如何安全开办学校的国家标准，因为总统说，“我没有责任。那不是我的回应。”我的意思是，字面上，这是他的短语。“这不是我的责任。这一切都不是我的错。让州长或市长们去处理它，”然后没有必要的财政援助来处理它。早在 7 月份，我就已经详细规划过，为了安全地开放，我们必须做些什么，包括为学校、学校老师和来学校的孩子们提供防护装备、口罩和其他需要的东西，手套等等。
> <| endoftext |>
> <| startoftext |>
> 没有任何理由，没有任何理由，为什么我们不能共同努力，我们不能再次共同努力克服这些挑战。你们都知道众议院通过了英雄法案，该法案将为地方和州政府提供 9150 亿美元。它现在正在参议院积灰。我很了解的多数党领袖

# **川普**生成了**的演讲**

> Thank you all. Thank you very much. Thank you very much. And I’m thrilled to be in North Carolina with thousands of hardworking American patriots who believe in faith, family, God, and country. Thank you. Thank you. It’s a great honor to be in the great state of North Carolina. Thank you. Thank you. It’s great to be in the state that I love. Thank you. And I’ll tell you what, you’re going to do it even better. You’re going to do it better. Thank you. We’re going to do it better. Look, we’re working hard. We’re going to do it better. You know that. We’re going to do it better. You know what it is? It’s a long time since we’ve done it. They had the worst year that they’ve had in 57 years. The Chinese Exclusion, what a terrible thing that was going on. It was going on when I took office. I mean, it was going on. Everybody knew it was going on. Democrats, Republicans, everybody. It was going on. It was going on in every community. It was going on in inner cities. It was going on in high taxes, high crime, open borders, late term abortion, and the worst debate performances in history. So I was very impressed by that. That was not the greatest. That was not the greatest. We did it last night. We’re doing it tonight. We’re doing it very well. It’s a great honor to be in the state with thousands of hardworking American patriots. Thank you. Thank you. It’s great to be in the state with thousands of proud, hardworking American patriots. Thank you, North Carolina. Thank you. It’s a great honor to be in North Carolina. Thank you. I appreciate it. What’s going on? You said it. Is there anything cooler than being in the room with the people that you love? Right? With all that we have going, I think you’ve got to say, I don’t know, it’s crazy. Crazy. You know what? I think you’ve got to give them a lot more credit than they deserve. That’s what it is. You know what? We’re in big numbers. We’re in big numbers. You know we’re in big numbers with China, with Russia, with fake news. You know what? If my people kept going in a straight line, it would have been seven fold higher. Take a look. I’ll be honest with you, Sean. I love Sean. I love him on television. I think he’s a great guy. But they turn the cameras around, and that’s where it’s going. That’s when it’s really going, because they know we’re going to turn this around. We’re going to be up to 138 miles an hour. We’re doing a route that we’re never done before. We’re doing the thing. It’s a great, great thing. It’s a great thing. Huge crowds all over the place. I think the crowds are now bigger than they’ve ever been in the history of this country. Believe me. We’re also joined tonight by some really great warriors. A man who’s a great fighter pilot and a great guy, Neil Armstrong. Neil Armstrong, you know Neil Armstrong, you know. You’re doing a great job, Neil. Neil Armstrong, you know. But he’s a great guy, and he loves this country. He’s a great, great fighter. And he is looking down with great dignity at these people for the first time. He’s looking down, and he’s saying things that you wouldn’t even believe. And they’re saying things that are unbelievable. We love these people. I just saw this guy, I don’t know if anybody knows this, but he’s been a friend of mine for a long time. He’s a very down and dirty Senator. This guy, Rand, he’s a great guy. But in the fight for freedom, there is no fight like the one we’re having right now. There is no fight like the one we’re having right now. I wonder why nobody called. I think we have the best fight in the world. I don’t know why anybody wouldn’t want to fight us. But nobody wants to fight Rand. That’s because he’s right. We’re They’re going to try and find something. There’s nothing like that. They’re going to look everywhere. Mike Bloomberg said, “I feel sorry for the children.” That’s what he said. “I feel sorry for the children.” I told Mike he’s wrong. I think the children are great. You know what the greatest thing is, they love our country. They love our country. The great state of Iowa. We love it. For years you watched as your politicians apologized for America. Now you have a president who is standing up for America and we are standing up for the people of Iowa. Thank you. And we will never ever stop fighting for the sacred values that bind us together as one America. We support, protect, and defend the constitution of the United States. We stand with the incredible heroes of law enforcement. We believe in the dignity of work and the sanctity of life. We believe that faith and family, not government and bureaucracy, are the true American way. And we believe that children should be taught to love our country, honor our history, and to always respect our great American flag. And we will always live by the words of our national motto, in God, we trust. We live by those words, in God, we trust. From Des Moines to Lincoln City, from Des Moines to now Reno, thanks to the incredible contributions of Iowa workers and citizens, we stand on the shoulders of true American heroes. Great people, great people. From Carson City to right here in Des Moines, we stand on the shoulders of true American heroes. Great people. Proud citizens like you helped build this country. And together we are taking back our country. We are returning power to you, the American people. With your help, your devotion, and your drive, we are going to keep on working. We are going to keep on fighting, and we are going to keep on winning, winning, winning. We are one movement, one people, one family, and one glorious nation under God. And together with the incredible people of Iowa, we will make America wealthy again. We will make America strong again. We will make America proud again. We will make America safe again. And we will make America great again. Thank you, Iowa. Thank you. Thank you.

# 最后

我们训练 GPT-2 生成新的演讲，生成的文本似乎与候选人的政治信仰几乎一致和相同。请记住，GPT-2 有“编造东西”的倾向，所以我们生成的结果仍然需要在 GPT-2 的未来实验中，用更多的语言处理，根据原始材料进行事实检查。