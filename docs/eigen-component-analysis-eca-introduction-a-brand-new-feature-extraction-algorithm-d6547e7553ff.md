# 特征分量分析(ECA)介绍——一种全新的特征提取算法

> 原文：<https://medium.com/analytics-vidhya/eigen-component-analysis-eca-introduction-a-brand-new-feature-extraction-algorithm-d6547e7553ff?source=collection_archive---------21----------------------->

![](img/25b73545ab1285f966eda81587d25049.png)

照片由 [Siora 摄影](https://unsplash.com/@siora18?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

这是 github 知识库对论文*本征分量分析的介绍:一种结合了机器学习技术的量子理论，用于寻找线性最大可分离分量。本实验包括两个主要部分，特征分量分析(ECA)和特征分量分析网络(ECAN)。无论是 ECA 还是 ECAN，都可以用标准的特征成分分析(VECA)或近似的特征成分分析(AECA)来训练。正如论文中提到的，VECA 经常导致稀疏结果，是降维的更好选择。*

依我的愚见，ECA 是一种一流的特征提取或降维算法。得到的特征矩阵(EFM)和特征类映射矩阵(ECMM)可以用来进行具体的降维。在没有背景或噪声的情况下，具体的降维数接近整个数据集的秩。此外，随着具体维数的减少，方差比接近 1。**例如，MNIST 数据集使用 VECA 的具体降维数是 110(使用 AECA 的是 328)，不多也不少。这可能意味着 MNIST 数据集或背景/无噪声数据集仅占据维度为 110 的子空间。VECA 和 AECA 结果的不同之处在于，VECA 忽略了一些不太重要的信息。**在 ECAN，利用维数算子，非线性降维优于许多经典算法，这将在我们未来的工作中报道。

核心算法在 *real_eigen.py* 中实现( *complex_eigen.py* 独立实现)。用于训练的基础模型以前缀 *base* 命名，可以独立运行，也可以作为一个模块工作。与 ECAN 相关的文件以*网络*为后缀。数据加载在 *load_data.py* 中实现。与其他模型的对比在 *other_models.py* 中实现。获得的 EFM、ECMM、RaDO 或 ReDO 存储在目录 *history 中。*

*   Analytic ECA: analytic_eca.py，可以找到满秩数据集的解析解
*   近似 ECA:base _ approximate py
*   复杂 ECA: complex_eigen.py，base_complex_eigen.py

分析 ECA 和基本模型中的所有 *data_tag* 都可以被改变以训练其他数据集。历史和检查点由*real _ eigen . py/complex _ eigen . py*中的 MAGIC_CODE 和每个待执行文件中的 WORK_MAGIC_CODE 管理。

在训练之前，应设置 EFM 和 ECMM 约束的两个超参数:

```
RealEigen.HP_ORTHONORMAL = 0.001
RealEigen.HP_EIGENDIST = 0.001
```

它们都可以被设置为较小的数目(根据经验`1e3`或更小),具有相对较大数目的训练时期。对于维数较低的数据集，这两个超参数取相对较大的值会加速收敛。放松对 EFM 的约束通常没有什么影响。然而，如果获得的 ECMM 不是二进制的，则相应的超参数应该设置得更大。一般来说，训练 VECA 比 AECA 容易，因为 ECMM 的二元性和稀疏性，这反映在 VECA 的惰性超参数设置上。

# 火车 VECA

*   这些文件包括 twodim.py ( `data_tag="2d"`)、threedim.py ( `data_tag="3d"`)、bc.py ( `data_tag="breast_cancer"`)、wis.py ( `data_tag="wis"`)、mnist.py ( `data_tag="mnist"`)对应的 2D、3D、Wis1992、Wis1995、mnist 数据集。
*   将 *to_train* 选项设置为 **True** ，否则将只在之前保存的模型上进行测试。
*   那么 Wis1992 上的培训应该是

> python bc.py

# 火车 AECA

*   可以改变 data_tag 来测试其他数据集。

> python base_approx.py

**在 MNIST 数据集上用 AECA 训练二折 ECAN**

*   ECAN 的代码在 base_network.py 中。可以将 data_tag 改为 load_data.py 中提到的那个
*   设置要使用的尺寸运算符

*   和 AECA 一起训练

*   将*设置为训练*为真**并在 MNIST 数据集上训练**

> python base_network.py

**在 MNIST 数据集上用 VECA 训练二折 ECAN**

*   与 AECA 训练的唯一区别在于这段代码

# 降维

在 history 文件夹中，有了对应的 MAGIC_CODE 和 WORK_MAGIC_CODE，我们就可以找到获得的 EFM *P* ，ECMM *LL* 。在 2 折 ECAN，EFM 和 ECMM 有后缀一个数字表示相应的折叠。RaDO 或 ReDO 都属于第一类。在 ECAN，单位运算符每隔一个折叠安装一次，因为一行中两个连续的维度运算符是微不足道的。

## 使用 ECA 降维

## 使用 2 重 ECAN 进行降维

降维运算符(重做)定义为

*原载于*[*http://github.com*](https://gist.github.com/fd45cea867dbe203f571268cd219788e)*。*

源代码:[https://github.com/chenmiaomiao/eca](https://github.com/chenmiaomiao/eca)

参考:

 [## 本征成分分析:一个量子理论结合机器学习技术，以寻找线性…

### 对于一个线性系统，对一个刺激的反应常常被它对其他分解刺激的反应所叠加。在…

arxiv.org](https://arxiv.org/abs/2003.10199) 

[https://www . researchgate . net/publication/340113956 _ Eigen _ component _ analysis _ A _ quantum _ theory _ incorporated _ machine _ technique _ to _ find _ linear _ maximum _ separable _ components](https://www.researchgate.net/publication/340113956_Eigen_component_analysis_A_quantum_theory_incorporated_machine_learning_technique_to_find_linearly_maximum_separable_components)