# 多层感知机 --- 从0开始

前面我们介绍了包括线性回归和多类逻辑回归的数个模型，它们的一个共同点是全是只含有一个输入层，一个输出层。这一节我们将介绍多层神经网络，就是包含至少一个隐含层的网络。

## 数据获取

我们继续使用FashionMNIST数据集。

```{.python .input  n=1}
import sys
sys.path.append('..')
import utils
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
```

## 多层感知机

多层感知机与前面介绍的[多类逻辑回归](../chapter_crashcourse/softmax-regression-scratch.md)非常类似，主要的区别是我们在输入层和输出层之间插入了一个到多个隐含层。

![](../img/multilayer-perceptron.png)

这里我们定义一个只有一个隐含层的模型，这个隐含层输出256个节点。

```{.python .input  n=2}
from mxnet import ndarray as nd

num_inputs = 28*28
num_outputs = 10

num_hidden = 256
weight_scale = .01

W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## 激活函数

如果我们就用线性操作符来构造多层神经网络，那么整个模型仍然只是一个线性函数。这是因为

$$\hat{y} = X \cdot W_1 \cdot W_2 = X \cdot W_3 $$

这里$W_3 = W_1 \cdot W_2$。为了让我们的模型可以拟合非线性函数，我们需要在层之间插入非线性的激活函数。这里我们使用ReLU

$$\textrm{rel}u(x)=\max(x, 0)$$

```{.python .input  n=3}
def relu(X):
    return nd.maximum(X, 0)
```

## 定义模型

我们的模型就是将层（全连接）和激活函数（Relu）串起来：

```{.python .input  n=4}
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
```

## Softmax和交叉熵损失函数

在多类Logistic回归里我们提到分开实现Softmax和交叉熵损失函数可能导致数值不稳定。这里我们直接使用Gluon提供的函数

```{.python .input  n=5}
from mxnet import gluon
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练

训练跟之前一样。

```{.python .input  n=10}
from mxnet import autograd as autograd

learning_rate = .5

for epoch in range(50):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0. Loss: 0.235895, Train acc 0.911892, Test acc 0.884315\nEpoch 1. Loss: 0.232244, Train acc 0.914129, Test acc 0.892328\nEpoch 2. Loss: 0.233658, Train acc 0.912827, Test acc 0.892127\nEpoch 3. Loss: 0.227683, Train acc 0.914497, Test acc 0.889022\nEpoch 4. Loss: 0.221442, Train acc 0.917234, Test acc 0.892929\nEpoch 5. Loss: 0.216925, Train acc 0.919838, Test acc 0.890525\nEpoch 6. Loss: 0.215313, Train acc 0.919888, Test acc 0.894832\nEpoch 7. Loss: 0.211224, Train acc 0.920573, Test acc 0.888822\nEpoch 8. Loss: 0.208956, Train acc 0.922710, Test acc 0.894331\nEpoch 9. Loss: 0.209033, Train acc 0.922409, Test acc 0.885517\nEpoch 10. Loss: 0.204946, Train acc 0.924179, Test acc 0.891326\nEpoch 11. Loss: 0.199053, Train acc 0.925564, Test acc 0.897436\nEpoch 12. Loss: 0.197816, Train acc 0.927334, Test acc 0.890024\nEpoch 13. Loss: 0.194410, Train acc 0.926649, Test acc 0.888622\nEpoch 14. Loss: 0.191025, Train acc 0.928803, Test acc 0.898137\nEpoch 15. Loss: 0.186017, Train acc 0.930372, Test acc 0.895433\nEpoch 16. Loss: 0.182889, Train acc 0.932559, Test acc 0.892929\nEpoch 17. Loss: 0.183630, Train acc 0.932592, Test acc 0.896234\nEpoch 18. Loss: 0.178219, Train acc 0.933644, Test acc 0.891226\nEpoch 19. Loss: 0.176907, Train acc 0.935280, Test acc 0.895733\nEpoch 20. Loss: 0.173927, Train acc 0.935513, Test acc 0.896234\nEpoch 21. Loss: 0.173260, Train acc 0.934963, Test acc 0.897937\nEpoch 22. Loss: 0.166209, Train acc 0.938268, Test acc 0.895032\nEpoch 23. Loss: 0.167631, Train acc 0.937183, Test acc 0.900240\nEpoch 24. Loss: 0.161374, Train acc 0.940421, Test acc 0.890024\nEpoch 25. Loss: 0.159970, Train acc 0.939804, Test acc 0.899038\nEpoch 26. Loss: 0.160268, Train acc 0.940772, Test acc 0.896234\nEpoch 27. Loss: 0.158090, Train acc 0.940722, Test acc 0.895833\nEpoch 28. Loss: 0.153476, Train acc 0.943209, Test acc 0.897436\nEpoch 29. Loss: 0.155652, Train acc 0.942358, Test acc 0.893429\nEpoch 30. Loss: 0.151357, Train acc 0.943576, Test acc 0.900040\nEpoch 31. Loss: 0.147631, Train acc 0.945446, Test acc 0.900140\nEpoch 32. Loss: 0.146896, Train acc 0.945596, Test acc 0.897336\nEpoch 33. Loss: 0.145169, Train acc 0.945463, Test acc 0.889724\nEpoch 34. Loss: 0.139580, Train acc 0.948768, Test acc 0.896434\nEpoch 35. Loss: 0.138722, Train acc 0.949336, Test acc 0.898838\nEpoch 36. Loss: 0.136971, Train acc 0.948902, Test acc 0.897236\nEpoch 37. Loss: 0.136582, Train acc 0.949052, Test acc 0.880509\nEpoch 38. Loss: 0.130141, Train acc 0.953626, Test acc 0.896635\nEpoch 39. Loss: 0.132462, Train acc 0.951372, Test acc 0.899840\nEpoch 40. Loss: 0.127439, Train acc 0.953843, Test acc 0.895733\nEpoch 41. Loss: 0.127242, Train acc 0.953759, Test acc 0.890024\nEpoch 42. Loss: 0.128722, Train acc 0.952340, Test acc 0.888421\nEpoch 43. Loss: 0.124959, Train acc 0.953726, Test acc 0.898638\nEpoch 44. Loss: 0.126712, Train acc 0.954193, Test acc 0.896434\nEpoch 45. Loss: 0.121821, Train acc 0.954861, Test acc 0.898438\nEpoch 46. Loss: 0.118952, Train acc 0.956697, Test acc 0.889523\nEpoch 47. Loss: 0.115509, Train acc 0.958183, Test acc 0.901743\nEpoch 48. Loss: 0.113632, Train acc 0.958417, Test acc 0.900140\nEpoch 49. Loss: 0.115513, Train acc 0.957983, Test acc 0.896234\n"
 }
]
```

## 总结

可以看到，加入一个隐含层后我们将精度提升了不少。

## 练习

- 我们使用了 `weight_scale` 来控制权重的初始化值大小，增大或者变小这个值会怎么样？
- 尝试改变 `num_hiddens` 来控制模型的复杂度
- 尝试加入一个新的隐含层

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/739)
