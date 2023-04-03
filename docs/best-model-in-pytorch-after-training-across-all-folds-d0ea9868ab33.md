# PyTorch 所有褶皱训练后的最佳模特

> 原文：<https://medium.com/analytics-vidhya/best-model-in-pytorch-after-training-across-all-folds-d0ea9868ab33?source=collection_archive---------11----------------------->

在本文中，我将定义一个函数，它将帮助社区在数据集的所有折叠中训练一个模型后保存最佳模型。

![](img/10182c3e624283378c05486a03acc469.png)

为此，首先我们将我们的数据帧划分成我们选择的若干折叠。

```
from sklearn import model_selection

dataframe["kfold"] = -1  ***# defining a new column in our dataset*** ***# taking a fraction of data from our dataframe and reset its index*** 
dataframe = dataframe.sample(frac=1).reset_index(drop=True)***# storing the target values in a list***
y = dataframe.label.values***# defining the number of required folds we want in our dataset***
kf = model_selection.StratifiedKFold(n_splits= ***number_of_folds***)***# assigning the fold number with respect to the data in kfold column*** 
for f, (t_, v_) **in** enumerate(kf.split(X=dataframe, y=y)):
    dataframe.loc[v_, 'kfold'] = f
**# looking into the new dataframe we created**     
dataframe
```

现在，我将定义一个函数，该函数将在所有折叠训练后返回最佳模型。

![](img/31deb4925d911840dbf2fa20de4f00d5.png)

```
from cpython.Lib import copya_string = "*" * 20def _run():

    for i **in** range(***number_of_folds***): ***# just a print statement for fomatting***
        print(a_string, " FOLD NUMBER ", i, a_string) 

        ***# assigning fold numbers for training dataframe***
        df_train = new_train[new_train.kfold != i].reset_index(drop=True) ***# assigning fold numbers for validation dataframe*** df_valid = new_train[new_train.kfold == i].reset_index(drop=True)

        ***# defining an empty list to store all accuracies across the particular fold we are training***
        all_accuracies = []

        for epoch **in** range(***NO_OF_EPOCHS***):
            print(f"Epoch --> **{**epoch+1**}** / **{*NO_OF_EPOCHS*}**")
            print(f"-------------------------------")

            train_loss, train_accuracy_metric_output = train_loop_fn()
            print('training Loss: **{:.4f}** & training accuracy : **{:.2f}**%'.format(train_loss, train_accuracy_metric_output*100))

            valid_loss , validation_accuracy_metric_output = eval_loop_fn(val_dataloader, model, device)
            print('validation Loss: **{:.4f}** & validation accuracy : **{:.2f}**%'.format(valid_loss , validation_accuracy_metric_output*100))

            all_accuracies.append(valid_acc)

        ***# comparing and saving the best model in best_model variable at each training and validation checkpoint***
        if i < 1:
            best_accuracy = max(all_accuracies)
            best_model = copy.deepcopy(model)
        else:
            if best_accuracy > max(all_accuracies):
                continue
            else:
                best_accuracy = max(all_accuracies)
                **best_model** = copy.deepcopy(model)

    print('**\n**============**Saving the best mode**l==================')
    torch.save(best_model.state_dict(),'__***your best model***__.pt')
    print() ***# Printing out the best accuracy we got after training***
    print("& The highest accuracy we got among all the folds is **{:.2f}**%".format(best_accuracy*100)) return best_model
```

由于上面的代码中有很多行，所以我将一个一个地解释它们，我也强烈要求阅读代码上面的注释行，以便正确地理解它。

在第**行的第**行，我从 cpython 库中导入复制函数。

然后定义一个字符串来格式化，这没什么大不了的。

然后定义一个循环，这样我们可以一个接一个地迭代每个折叠。

之后，我们使用格式化变量来显示以迭代方式在每个褶皱上训练模型所获得的结果。

然后定义 df_train & df_valid，它将只考虑 df_valid 的折叠数，其余的折叠数将分配给 df_train。

在此之后，我们定义一个空列表来存储所有的精度值，这些值是我们在验证数据的特定折叠数内训练模型后获得的。

之后，我们定义循环来为多个时期训练我们的模型，并使用一些格式化打印行来跟踪训练和验证损失，以及度量并在接下来的几行中打印出来。

**all _ accurances . append(valid _ ACC)**该行用于存储上述列表中折叠训练内获得的所有精度。

接下来的几行是最重要的几行，实际的魔术正在发生，所以请密切注意这一点。 ***因此，我们在这里检查，如果我们当前训练模型的折叠小于 1，那么我们将在两个变量 best_accuracy & best_model 中连续存储最佳精度和最佳模型。否则，如果折叠数的值大于或等于 1，那么我们将它的最佳精度与我们在上一次折叠迭代中获得的最大精度进行比较。如果当前的最佳精度小于我们之前得到的精度，那么我们只是继续，否则我们用我们需要的变量中的值替换它。***

我希望您已经很好地理解了 pytorch 代码。如果你还有任何问题、意见或顾虑，请不要害羞，在评论区提出来。我一定会回答的。在那之前享受学习。