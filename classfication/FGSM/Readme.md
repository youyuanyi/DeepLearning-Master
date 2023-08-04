# FGSM代码说明

## 组成

- train.py：以MNIST数据集为例，用的是LeNet
- model.py：LeNet
- resnet_fgsm.py：以resnet18为例，用的是预训练权重，重新训练1个epoch
- ex_resnet.py：以resnet34为例，数据集为Fashion-MNIST，代码封装性比较强

其他的不需要关注了

## FGSM伪代码

1.输入：原始图像`data`，对应的标签`target_label`，模型`model`

2.将`data`设置为需要梯度跟踪：`data.requires_grad = True`。这样才能获取输入的梯度

3.将`data`输入`model`进行前向传播计算：`output = model(data)`

4.计算损失函数：`loss = loss_function(output,target_label)`

5.将`model`的所有参数梯度清零：`model.zero_grad()`

6.反向传播计算梯度：`loss.backward()`。这里的梯度就包括关于`data`的梯度

7.使用得到的梯度计算`扰动`：`perturbation = epsilon*torch.sign(data.grad.data)`

8.将`扰动`加入输入`data`中，得到`对抗样本`：`perturbed_data = data+perturbation`

9.将对抗样本输入模型进行预测，测试攻击效果：`perturbed_output = model(perturbed_data)`



## 为什么FGSM中使用的是model.zero_grad()而不是optimizer.zero_grad()

- `optimizer.zero_grad()`用来清空优化器中已注册参数的梯度信息。这是因为在训练过程中，我们的目标是通过反向传播算法计算损失函数相对于模型参数的梯度，然后**使用优化器来更新这些参数，从而最小化损失函数**
- 在FGSM算法中，我们目标不是更新模型参数，而是产生对抗样本。
- 使用`model.zero_grad()`是为了确保在生成对抗样本时，不会计算对模型参数的梯度，我们只需要输入数据的梯度就行。
- 如果在 FGSM 中使用 `optimizer.zero_grad()`，将会清空优化器中的参数梯度，而这并不是我们所需要的，因为我们不是在训练模型，而是在攻击模型



## 待解决

### fgsm中epsilon=0

```python
# fgsm产生对抗样本
def fgsm_attack(image, epsilon, data_grad):
    # 收集数据梯度的元素符号
    sign_data_grad = data_grad.sign()
    # 通过调整输入图像的每个像素来创建扰动图像
    perturbed_image = image + epsilon * sign_data_grad
    # 添加剪切以维持[0,1]范围,仅操作epsilon!=0的情况
    if epsilon != 0:
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回被扰动的图像
    return perturbed_image
```

在这段fgsm代码中，如果epsilon=0时也对生成的对抗样本限制在0-1范围内，那么当epsilon=0时，模型准确率只有50-60%左右，去掉后恢复到93%

对于其他>0的epsilon值，没有区别



### resnet34随着训练epoch增加，fgsm对抗攻击更加明显（即准确率更低）。但是在测试集上的准确率是正常的

猜测是resnet34可能太深了，对于Fashion-MNIST不太合适，有点过拟合了（一点扰动就会造成较大的错误率）。因为用resnet18做出来就是正常的。
