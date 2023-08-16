## Pytorch实现CW攻击

### 总体说明

流程和PGD差不多，建议先看懂我PGD对抗训练那部分的代码

### 各文件功能

- main.py：以ImageNet-1000为数据集，Inception v3为预训练权重进行推测，并加入CW攻击测试效果
- read_data.py：读取CIFAR-10数据集
- train.py：训练代码
- util.py：实现了CW攻击类和对抗训练的代码（和PGD差不多，只是攻击方式不一样）

### 训练效果

以CIFAR-10数据集，wrn16_10为模型进行CW攻击对抗训练，基本上无法训练成功，准确率在10%左右，证明CW攻击十分有效

### 推理效果

#### main.py的结果

##### 干净图片

<img src=".\干净图片.png" alt="干净图片" style="zoom: 50%;" />

##### 对抗图片

<img src=".\CW攻击的图片.png" alt="CW攻击的图片" style="zoom:50%;" />

### 其他

test.py没有修改，是PGD版本的，建议要推理的话用main.py，只要把测试数据打包成loader后，加入扰动就行了