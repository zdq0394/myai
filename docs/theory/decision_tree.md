# 决策树
## 概述
* 归纳推理算法之一。
* 逼近离散值函数的方法。
* 广为应用的算法包括：
    * ID3
    * C4.5
    * c5.0
    * CART

通常，决策树学习算法适合有以下特征的问题：
* 实例是“属性-值”对表示的。
* 目标函数具有离散的输出值。
* 可能需要析取的描述。
* 训练数据可以包含错误。
* 训练数据可以包含缺少属性值的实例。

ID3的搜索策略为：
1. 优先选择较短的树而不是较长的；
2. 选择那些信息增益高的属性离根节点较近的树。

决策树学习的常见问题：
* 确定决策树增长的深度
* 处理连续值的属性
* 选择一个适当的属性筛选度量标准
* 处理属性值不完整的训练数据
* 处理不同代价的属性
* 提高计算效率

## 概念
决策树是一种解决分类问题的算法，决策树算法采用树形结构，使用层层推理来实现最终的分类。决策树由下面几种元素构成：
* 根节点：包含样本的全集
* 内部节点：对应特征属性测试
* 叶节点：代表决策的结果

## 决策树学习的三个步骤
1. 特征选择：在特征选择中通常使用的准则是：信息增益。
2. 决策树生成
3. 决策树剪枝：剪枝的主要目的是对抗“过拟合”，通过主动去掉部分分支来降低过拟合的风险。

## 典型的决策树算法
* ID3：最早提出的决策树算法，利用信息增益来选择特征。
* C4.5：ID3的改进版，不是直接使用信息增益，而是引入“信息增益比”指标作为特征的选择依据。
* CART：使用基尼系数取代信息熵模型。这种算法即可以用于分类，也可以用于回归问题。