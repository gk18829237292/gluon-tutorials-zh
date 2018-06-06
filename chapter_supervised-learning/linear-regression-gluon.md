# 线性回归 --- 使用Gluon

[前一章](linear-regression-scratch.md)我们仅仅使用了`ndarray`和`autograd`来实现线性回归，这一章我们仍然实现同样的模型，但是使用高层抽象包`gluon`。

## 创建数据集

我们生成同样的数据集

```{.python .input  n=35}
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)
```

## 数据读取

但这里使用`data`模块来读取数据。

```{.python .input  n=36}
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
```

读取跟前面一致：

```{.python .input  n=37}
for data, label in data_iter:
    print(data, label)
    break
```

```{.json .output n=37}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[-0.96869022 -0.80516362]\n [ 0.34829465 -1.18665373]\n [-1.41899419  0.66632795]\n [ 1.67505109 -0.5115416 ]\n [-1.74669123 -0.09182929]\n [ 0.02645121 -1.5654459 ]\n [ 0.31733912  1.36049843]\n [ 1.33297002 -1.05613756]\n [ 1.79389966  0.2958298 ]\n [ 0.38759562 -0.98514628]]\n<NDArray 10x2 @cpu(0)> \n[  4.99481964   8.93494892  -0.90762198   9.2727623    1.00991619\n   9.56194687   0.20353241  10.46225548   6.78135777   8.31671906]\n<NDArray 10 @cpu(0)>\n"
 }
]
```

## 定义模型

之前一章中，当我们从0开始训练模型时，需要先声明模型参数，然后再使用它们来构建模型。但`gluon`提供大量预定义的层，我们只需要关注使用哪些层来构建模型。例如线性模型就是使用对应的`Dense`层；之所以称为dense层，是因为输入的所有节点都与后续的节点相连。在这个例子中仅有一个输出，但在大多数后续章节中，我们会用到具有多个输出的网络。

我们之后还会介绍如何构造任意结构的神经网络，但对于初学者来说，构建模型最简单的办法是利用`Sequential`来所有层串起来。输入数据之后，`Sequential`会依次执行每一层，并将前一层的输出，作为输入提供给后面的层。首先我们定义一个空的模型：

```{.python .input  n=38}
net = gluon.nn.Sequential()
```

然后我们加入一个`Dense`层，它唯一必须定义的参数就是输出节点的个数，在线性模型里面是1.

```{.python .input  n=39}
net.add(gluon.nn.Dense(1))
```

```{.python .input  n=40}
net
```

```{.json .output n=40}
[
 {
  "data": {
   "text/plain": "Sequential(\n  (0): Dense(None -> 1, linear)\n)"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

（注意这里我们并没有定义说这个层的输入节点是多少，这个在之后真正给数据的时候系统会自动赋值。我们之后会详细介绍这个特性是如何工作的。）

## 初始化模型参数

在使用前`net`我们必须要初始化模型权重，这里我们使用默认随机初始化方法（之后我们会介绍更多的初始化方法）。

```{.python .input  n=41}
net.initialize()
```

## 损失函数

`gluon`提供了平方误差函数：

```{.python .input  n=42}
square_loss = gluon.loss.L2Loss()
```

## 优化

同样我们无需手动实现随机梯度下降，我们可以创建一个`Trainer`的实例，并且将模型参数传递给它就行。

```{.python .input  n=50}
trainer = gluon.Trainer(
    net.collect_params(), 'sgd', {'learning_rate': 0.1})
```

## 训练
使用`gluon`使模型训练过程更为简洁。我们不需要挨个定义相关参数、损失函数，也不需使用随机梯度下降。`gluon`的抽象和便利的优势将随着我们着手处理更多复杂模型的愈发显现。不过在完成初始设置后，训练过程本身和前面没有太多区别，唯一的不同在于我们不再是调用`SGD`，而是`trainer.step`来更新模型（此处一并省略之前绘制损失变化的折线图和散点图的过程，有兴趣的同学可以自行尝试）。

```{.python .input  n=51}
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
```

```{.json .output n=51}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0, average loss: 0.000085\nEpoch 1, average loss: 0.000050\nEpoch 2, average loss: 0.000050\nEpoch 3, average loss: 0.000050\nEpoch 4, average loss: 0.000050\n"
 }
]
```

比较学到的和真实模型。我们先从`net`拿到需要的层，然后访问其权重和位移。

```{.python .input  n=45}
dense = net[0]
true_w, dense.weight.data()
```

```{.json .output n=45}
[
 {
  "data": {
   "text/plain": "([2, -3.4], \n [[ 1.98630238 -3.38178253]]\n <NDArray 1x2 @cpu(0)>)"
  },
  "execution_count": 45,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=46}
true_b, dense.bias.data()
```

```{.json .output n=46}
[
 {
  "data": {
   "text/plain": "(4.2, \n [ 4.17264318]\n <NDArray 1 @cpu(0)>)"
  },
  "execution_count": 46,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=49}
help(trainer.step)
```

```{.json .output n=49}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on method step in module mxnet.gluon.trainer:\n\nstep(batch_size, ignore_stale_grad=False) method of mxnet.gluon.trainer.Trainer instance\n    Makes one step of parameter update. Should be called after\n    `autograd.compute_gradient` and outside of `record()` scope.\n    \n    Parameters\n    ----------\n    batch_size : int\n        Batch size of data processed. Gradient will be normalized by `1/batch_size`.\n        Set this to 1 if you normalized loss manually with `loss = mean(loss)`.\n    ignore_stale_grad : bool, optional, default=False\n        If true, ignores Parameters with stale gradient (gradient that has not\n        been updated by `backward` after last step) and skip update.\n\n"
 }
]
```

```{.python .input  n=48}
help(dense.weight)
```

```{.json .output n=48}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on Parameter in module mxnet.gluon.parameter object:\n\nclass Parameter(builtins.object)\n |  A Container holding parameters (weights) of Blocks.\n |  \n |  :py:class:`Parameter` holds a copy of the parameter on each :py:class:`Context` after\n |  it is initialized with ``Parameter.initialize(...)``. If :py:attr:`grad_req` is\n |  not ``'null'``, it will also hold a gradient array on each :py:class:`Context`::\n |  \n |      ctx = mx.gpu(0)\n |      x = mx.nd.zeros((16, 100), ctx=ctx)\n |      w = mx.gluon.Parameter('fc_weight', shape=(64, 100), init=mx.init.Xavier())\n |      b = mx.gluon.Parameter('fc_bias', shape=(64,), init=mx.init.Zero())\n |      w.initialize(ctx=ctx)\n |      b.initialize(ctx=ctx)\n |      out = mx.nd.FullyConnected(x, w.data(ctx), b.data(ctx), num_hidden=64)\n |  \n |  Parameters\n |  ----------\n |  name : str\n |      Name of this parameter.\n |  grad_req : {'write', 'add', 'null'}, default 'write'\n |      Specifies how to update gradient to grad arrays.\n |  \n |      - ``'write'`` means everytime gradient is written to grad :py:class:`NDArray`.\n |      - ``'add'`` means everytime gradient is added to the grad :py:class:`NDArray`. You need\n |        to manually call ``zero_grad()`` to clear the gradient buffer before each\n |        iteration when using this option.\n |      - 'null' means gradient is not requested for this parameter. gradient arrays\n |        will not be allocated.\n |  shape : tuple of int, default None\n |      Shape of this parameter. By default shape is not specified. Parameter with\n |      unknown shape can be used for :py:class:`Symbol` API, but ``init`` will throw an error\n |      when using :py:class:`NDArray` API.\n |  dtype : numpy.dtype or str, default 'float32'\n |      Data type of this parameter. For example, ``numpy.float32`` or ``'float32'``.\n |  lr_mult : float, default 1.0\n |      Learning rate multiplier. Learning rate will be multiplied by lr_mult\n |      when updating this parameter with optimizer.\n |  wd_mult : float, default 1.0\n |      Weight decay multiplier (L2 regularizer coefficient). Works similar to lr_mult.\n |  init : Initializer, default None\n |      Initializer of this parameter. Will use the global initializer by default.\n |  \n |  Attributes\n |  ----------\n |  grad_req : {'write', 'add', 'null'}\n |      This can be set before or after initialization. Setting ``grad_req`` to ``'null'``\n |      with ``x.grad_req = 'null'`` saves memory and computation when you don't\n |      need gradient w.r.t x.\n |  lr_mult : float\n |      Local learning rate multiplier for this Parameter. The actual learning rate\n |      is calculated with ``learning_rate * lr_mult``. You can set it with\n |      ``param.lr_mult = 2.0``\n |  wd_mult : float\n |      Local weight decay multiplier for this Parameter.\n |  \n |  Methods defined here:\n |  \n |  __init__(self, name, grad_req='write', shape=None, dtype=<class 'numpy.float32'>, lr_mult=1.0, wd_mult=1.0, init=None, allow_deferred_init=False, differentiable=True)\n |      Initialize self.  See help(type(self)) for accurate signature.\n |  \n |  __repr__(self)\n |      Return repr(self).\n |  \n |  cast(self, dtype)\n |      Cast data and gradient of this Parameter to a new data type.\n |      \n |      Parameters\n |      ----------\n |      dtype : str or numpy.dtype\n |          The new data type.\n |  \n |  data(self, ctx=None)\n |      Returns a copy of this parameter on one context. Must have been\n |      initialized on this context before.\n |      \n |      Parameters\n |      ----------\n |      ctx : Context\n |          Desired context.\n |      \n |      Returns\n |      -------\n |      NDArray on ctx\n |  \n |  grad(self, ctx=None)\n |      Returns a gradient buffer for this parameter on one context.\n |      \n |      Parameters\n |      ----------\n |      ctx : Context\n |          Desired context.\n |  \n |  initialize(self, init=None, ctx=None, default_init=<mxnet.initializer.Uniform object at 0x7f105bab1208>, force_reinit=False)\n |      Initializes parameter and gradient arrays. Only used for :py:class:`NDArray` API.\n |      \n |      Parameters\n |      ----------\n |      init : Initializer\n |          The initializer to use. Overrides :py:meth:`Parameter.init` and default_init.\n |      ctx : Context or list of Context, defaults to :py:meth:`context.current_context()`.\n |          Initialize Parameter on given context. If ctx is a list of Context, a\n |          copy will be made for each context.\n |      \n |          .. note::\n |              Copies are independent arrays. User is responsible for keeping\n |              their values consistent when updating.\n |              Normally :py:class:`gluon.Trainer` does this for you.\n |      \n |      default_init : Initializer\n |          Default initializer is used when both :py:func:`init`\n |          and :py:meth:`Parameter.init` are ``None``.\n |      force_reinit : bool, default False\n |          Whether to force re-initialization if parameter is already initialized.\n |      \n |      Examples\n |      --------\n |      >>> weight = mx.gluon.Parameter('weight', shape=(2, 2))\n |      >>> weight.initialize(ctx=mx.cpu(0))\n |      >>> weight.data()\n |      [[-0.01068833  0.01729892]\n |       [ 0.02042518 -0.01618656]]\n |      <NDArray 2x2 @cpu(0)>\n |      >>> weight.grad()\n |      [[ 0.  0.]\n |       [ 0.  0.]]\n |      <NDArray 2x2 @cpu(0)>\n |      >>> weight.initialize(ctx=[mx.gpu(0), mx.gpu(1)])\n |      >>> weight.data(mx.gpu(0))\n |      [[-0.00873779 -0.02834515]\n |       [ 0.05484822 -0.06206018]]\n |      <NDArray 2x2 @gpu(0)>\n |      >>> weight.data(mx.gpu(1))\n |      [[-0.00873779 -0.02834515]\n |       [ 0.05484822 -0.06206018]]\n |      <NDArray 2x2 @gpu(1)>\n |  \n |  list_ctx(self)\n |      Returns a list of contexts this parameter is initialized on.\n |  \n |  list_data(self)\n |      Returns copies of this parameter on all contexts, in the same order\n |      as creation.\n |  \n |  list_grad(self)\n |      Returns gradient buffers on all contexts, in the same order\n |      as :py:meth:`values`.\n |  \n |  reset_ctx(self, ctx)\n |      Re-assign Parameter to other contexts.\n |      \n |      ctx : Context or list of Context, default ``context.current_context()``.\n |          Assign Parameter to given context. If ctx is a list of Context, a\n |          copy will be made for each context.\n |  \n |  set_data(self, data)\n |      Sets this parameter's value on all contexts.\n |  \n |  var(self)\n |      Returns a symbol representing this parameter.\n |  \n |  zero_grad(self)\n |      Sets gradient buffer on all contexts to 0. No action is taken if\n |      parameter is uninitialized or doesn't require gradient.\n |  \n |  ----------------------------------------------------------------------\n |  Data descriptors defined here:\n |  \n |  __dict__\n |      dictionary for instance variables (if defined)\n |  \n |  __weakref__\n |      list of weak references to the object (if defined)\n |  \n |  grad_req\n |  \n |  shape\n\n"
 }
]
```

## 结论

可以看到`gluon`可以帮助我们更快更干净地实现模型。


## 练习

- 在训练的时候，为什么我们用了比前面要大10倍的学习率呢？（提示：可以尝试运行 `help(trainer.step)`来寻找答案。）
- 如何拿到`weight`的梯度呢？（提示：尝试 `help(dense.weight)`）

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/742)
