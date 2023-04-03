# 平衡的艺术(主厨二月长)

> 原文：<https://medium.com/analytics-vidhya/art-of-balance-codechef-february-long-190e19ea1dec?source=collection_archive---------16----------------------->

这个问题是在 2 月份的 codechef long 挑战赛中问的。问题是这样的:

“宇宙中的一切都是平衡的。你在生活中面临的每一次失望都会被对你有利的东西所平衡！坚持下去，永不放弃。”

如果一个字符串中出现的所有字符在其中出现的次数相同，那么我们称这个字符串为平衡的。

给你一个字符串 S；该字符串只能包含大写英文字母。您可以执行以下操作任意次(包括零次):选择 S 中的一个字母，并用另一个大写英文字母替换它。请注意，即使被替换的字母在 S 中出现多次，也只会替换该字母中被选中的那个。

找出将给定字符串转换为平衡字符串所需的最少运算次数。

![](img/50fb68cb1bf8171408bbd1eba908d61d.png)

起初，我不知道如何解决这个问题。但是让我们一步一步地解码

1.  该字符串可以由单个字符组成，也可以由英语字母表的所有 26 个字符组成

```
We need to have a loop that will go from 26 upto 1 and find the number of replacement needed in each case
```

2.如果输出字符串(在替换和转换之后)具有“**I”**个不同的字符，那么字符串的长度 **l** 模数 **i** 将为零。即“**l % I = 0”**

```
We can ignore cases while looping if l%i !=0
```

3.如果我们考虑" **i"** 字符串中截然不同的字符(final)，那么每个字符的出现频率将是" **l/i"**

```
We need to have a count array that will store frequency of each character
```

4.最坏情况下的最大运算将是(length-1)，其中 length 表示字符串的长度

```
Initial result as length-1
```

有了这 4 点认识之后，编写这个问题就非常简单了。下面是整个问题的代码

```
#include <bits/stdc++.h>
using namespace std;
int main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    int t;
    cin>>t;
    while(t--){
        string s;
        cin>>s;
        long int len = s.length(); /* Max operations needed in worst case */ long int operations = len-1; /* To maintain frequency */ long int count[26]={0};
        for(long int i=0;i<len;i++){
            count[s[i]-'A']++;
        } /* Why sort? Explained in comment below*/ sort(count,count+26); /* Consider that the ouput string can either have all 26 character or just a single character */ for(int i=26;i>=1;i--){ /* Consider a string having i distinct character only if it satisfies below criteria */ if(len%i==0){
                long int temp=0;
                long int frequency = len/i; /* Considering that each character must appear frequency times, subtracting from initial count to make frequency same*/ for(int j=25;j>=26-i;j--){
                    temp+= abs(frequency-count[j]);
                }
                /* Final output string must have only i characters, rest of character count must be zero, reason why we sorted the count array initially */
                for(int j=25-i;j>=0;j--){
                    if(count[j]!=0){
                        temp+= count[j];
                    }
                }
                /* Dividing by 2 because changing a to e (example only) will increase count of e and decrease count of a at same time */
                temp/=2;
                if(operations>temp){
                    operations = temp;
                }
            }
        }
        cout<<operations<<"\n";
    }   
    return 0;
}
```