机器学习和人类学习有着千丝万缕的联系，人类试图构建一些学会学习（Learning to learn）的方式教给电脑/算法以学习人类的学习方法。 通常很多业务场景都有一些模式可以被识别出来发现事物之间的关联和区别，有经验的人类也是可以完成这些目标的。
但是有时任务内的关系可能太复杂（比如大量图片和抽象类别之间的关系），需要数千或数百万次的计算。 即使人类的眼睛能毫不费力地完成这些难以提出完美解决方案的任务，这其中的计算也超出了人类意识理解范畴。 机器学习/深度学习是一类强大的可以从经验中学习的技术。 


一个经典的机器学习任务包括以下流程：
1. 数据（特征工程，是否有label）
	a. 数据怎么来：爬虫/公开数据集/开放数据接口等
	b. 数据太少/数据不均衡/异常值/缺失值
	c. 数据规约/数据变换/提取特征
	d. Why？
2. 算法模型
	a. 如何选取模型算法，特征筛选，模型比较，AIC/BIC
	b. 输入输出是什么
3. 目标函数
	a. 准确率，MLE，Loss，奖励函数，F1等
	b. 过拟合/欠拟合/过参/正则化
4. 优化算法
	a. Adam/SGD等

![image](https://github.com/user-attachments/assets/b2101627-fbdb-4936-8bde-cec839671003)


# 2.1 有监督学习
在分类任务中，标签都是离散值；而在回归任务中，标签都是连续值。
## 2.1.1 分类算法（Regression）
### 逻辑回归/Softmax回归
- 两类别模型
本质上就是把一个线性回归函数 $\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b$ 映射为一个概率值，最理想的就是一个阶梯函数，但是不能优化，所以sigmoid取代。

逻辑回归的本质就是 $y(\boldsymbol{x})=\sigma\left(\boldsymbol{w}^T \boldsymbol{x}\right)$，即此时公式为

$$y=\frac{1}{1+e^{-\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right)}} $$


<!-- ##{"script":"<script src='https://blog.meekdai.com/Gmeek/plugins/GmeekTOC.js'></script>"}## -->
