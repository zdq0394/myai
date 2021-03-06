# 概述
## 现代推荐架构剖析
推荐架构需要解决的问题：
1. 能够在一两百毫秒之内给用户提供当前的推荐结果；
2. 对用户和系统的交互结果做出相应；
3. 考虑用户群体的覆盖率问题。
### 基于线下离线计算的推荐架构
线下离线计算的一个主要想法就是：把计算中复杂的步骤尽量提前做好，然后当用户来到网站需要呈现结果的时候，我们要么已经完成了所有的计算，要么还剩非常少的步骤可以在很快的时间内，也就是说一两百毫秒之内完成剩下的计算。（可以解决问题1）

### 基于多层搜索架构的推荐系统
* 多层搜索架构可以支持搜索结果，自然地，对实时呈现推荐结果有很少的支持。
* 需要对用户的反馈进行更新，可以在重排序的阶段，通过两种方式实现：
    * 更新重新排序的模型
    * 更新重排序模型的某些特性
* 搜索架构对新的用户是天然支持的，但是对新的物品的支持是短板。

### 复杂现代推荐架构
当我们面对新用户多和新物品多的场景时，推荐架构的一些基本原则：
* 尽可能把复杂的运算放在线下，因为毕竟需要在规定的时间内返回结果；
* 在一切有可能的情况下，尽可能使用搜索引擎来减少需要对大量物品进行打分的步骤；
* 对于活跃的用户，我们可以使用多层搜索架构；但是对于不活跃的用户，我们可以依赖线下，提前产生所有的推荐结果。

## 简单推荐模型
### 基于流行度的推荐模型
* 物品流行度影响因素：时间和位置；
* 对于流行度的度量，我们往往使用的是一个“比值”，或者是计算某种“可能性”；
* 建立无偏差的数据：epsilon贪心、最大似然估计法。

### 基于相似信息的推荐模型
相似信息的推荐模型又称为“临近”模型，其内在假设是“协同过滤”：
* 相似的用户可能会有相似的喜好
* 相似的物品可能被相似的人所偏好

“协同过滤”从统计模型的意义上来讲，其实就是“借用数据”，在数据稀缺的情况下帮助建模。

### 基于内容信息的推荐模型
基于内容信息的推荐系统，其实就是用特征来表示用户、物品以及用户和物品的交互，从而能够把推荐问题转换成监督学习任务，有两个关键步骤：
* 特征工程
* 目标函数

内容信息的各类特性：
* 物品的文本信息
* 物品的类别信息（或者物品的知识信息）
* 用户的基本特性，包括性别、年龄、地理位置；
* 用户画像。

## 基于隐变量的模型
### 矩阵分解

### 基于回归的矩阵分解

### 分解机

## 高阶推荐模型
### 张量分解模型
### 协同矩阵分解
### 优化复杂目标函数

## 推荐的EE算法
### Exploit和Explore算法
EE可以看作是一个优化过程，需要多次迭代才能找到比较好的方案。
EE的产品部署有两大难点：如何上线测试、如何平衡产品。

### UCB算法
UCB算法本身，其实是同时考虑了物品现在的情况以及在这种情况下的置信度，并且寄希望通过多次迭代来达到减小标准差，提高置信度的目的。

UCB算法本质上还是“确定性”算法，并没有随机性。

### 汤普森采样算法
核心要点：
* 每一轮，汤普森采样都有一个参数采样的动作；从后验概率分布中进行抽样；
* 因为使用了贝叶斯统计，对参数有先验的设置，因此针对当前点击率估计还不准确甚至还没有数据的物品来说，有天然的优势；
* 因为是采样，即便是在参数一样的情况下，两个物品的采样数值都有可能是不一样的，一举解决了UCB问题。

## 基于深度学习的推荐模型
### 受限波兹曼机
### 基于RNN的推荐系统
### 利用深度学习来扩展推荐系统

## 推荐系统的评价
### 传统线下评测
### 线上评测
* 用户的驻留时间或者停留时间
* 用户在相邻两次访问中的间隔时间，有时叫做“空缺时间”
### 无偏差估计
