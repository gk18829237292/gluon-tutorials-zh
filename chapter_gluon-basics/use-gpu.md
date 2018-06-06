# 使用GPU来计算

【注意】运行本教程需要GPU。没有GPU的同学可以大致理解下内容，至少是`context`这个概念，因为之后我们也会用到。但没有GPU不会影响运行之后的大部分教程（好吧，还是有点点，可能运行会稍微慢点）。

前面的教程里我们一直在使用CPU来计算，因为绝大部分的计算设备都有CPU。但CPU的设计目的是处理通用的计算，例如打开浏览器和运行Jupyter，它一般只有少数的一块区域复杂数值计算，例如`nd.dot(A, B)`。对于复杂的神经网络和大规模的数据来说，单块CPU可能不够给力。

常用的解决办法是要么使用多台机器来协同计算，要么使用数值计算更加强劲的硬件，或者两者一起使用。本教程关注使用单块Nvidia GPU来加速计算，更多的选项例如多GPU和多机器计算则留到后面。

首先需要确保至少有一块Nvidia显卡已经安装好了，然后下载安装显卡驱动和[CUDA](https://developer.nvidia.com/cuda-downloads)（推荐下载8.0，CUDA自带了驱动）。完成后应该可以通过`nvidia-smi`查看显卡信息了。（Windows用户需要设一下PATH：`set PATH=C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%`）。

```{.python .input  n=1}
!nvidia-smi
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sat Jun  2 10:26:56 2018       \r\n+-----------------------------------------------------------------------------+\r\n| NVIDIA-SMI 390.25                 Driver Version: 390.25                    |\r\n|-------------------------------+----------------------+----------------------+\r\n| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\r\n| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\r\n|===============================+======================+======================|\r\n|   0  GeForce GTX 106...  Off  | 00000000:23:00.0  On |                  N/A |\r\n|  0%   46C    P8    10W / 120W |    276MiB /  6077MiB |      0%      Default |\r\n+-------------------------------+----------------------+----------------------+\r\n                                                                               \r\n+-----------------------------------------------------------------------------+\r\n| Processes:                                                       GPU Memory |\r\n|  GPU       PID   Type   Process name                             Usage      |\r\n|=============================================================================|\r\n|    0      1342      G   /usr/lib/xorg/Xorg                           183MiB |\r\n|    0      2714      G   compiz                                        90MiB |\r\n+-----------------------------------------------------------------------------+\r\n"
 }
]
```

接下来要要确认正确安装了的`mxnet`的GPU版本。具体来说是卸载了`mxnet`（`pip uninstall mxnet`），然后根据CUDA版本安装`mxnet-cu75`或者`mxnet-cu80`（例如`pip install --pre mxnet-cu80`）。

使用pip来确认下：

```{.python .input  n=19}
import pip
for pkg in ['mxnet', 'mxnet-cu75', 'mxnet-cu80','mxnet-cu90']:
    pip.main(['show', pkg])
```

## Context

MXNet使用Context来指定使用哪个设备来存储和计算。默认会将数据开在主内存，然后利用CPU来计算，这个由`mx.cpu()`来表示。GPU则由`mx.gpu()`来表示。注意`mx.cpu()`表示所有的物理CPU和内存，意味着计算上会尽量使用多有的CPU核。但`mx.gpu()`只代表一块显卡和其对应的显卡内存。如果有多块GPU，我们用`mx.gpu(i)`来表示第*i*块GPU（*i*从0开始）。

```{.python .input  n=20}
import mxnet as mx
[mx.cpu(), mx.gpu(), mx.gpu(2)]
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "[cpu(0), gpu(0), gpu(2)]"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## NDArray的GPU计算

每个NDArray都有一个`context`属性来表示它存在哪个设备上，默认会是`cpu`。这是为什么前面每次我们打印NDArray的时候都会看到`@cpu(0)`这个标识。

```{.python .input  n=21}
from mxnet import nd
x = nd.array([1,2,3])
x.context
```

```{.json .output n=21}
[
 {
  "data": {
   "text/plain": "cpu(0)"
  },
  "execution_count": 21,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### GPU上创建内存

我们可以在创建的时候指定创建在哪个设备上（如果GPU不能用或者没有装MXNet GPU版本，这里会有error）：

```{.python .input  n=22}
a = nd.array([1,2,3], ctx=mx.gpu())
b = nd.zeros((3,2), ctx=mx.gpu())
c = nd.random.uniform(shape=(2,3), ctx=mx.gpu())
(a,b,c)
```

```{.json .output n=22}
[
 {
  "data": {
   "text/plain": "(\n [ 1.  2.  3.]\n <NDArray 3 @gpu(0)>, \n [[ 0.  0.]\n  [ 0.  0.]\n  [ 0.  0.]]\n <NDArray 3x2 @gpu(0)>, \n [[ 0.70019829  0.17011374  0.60112673]\n  [ 0.95502114  0.26806629  0.62099582]]\n <NDArray 2x3 @gpu(0)>)"
  },
  "execution_count": 22,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

尝试将内存开到另外一块GPU上。如果不存在会报错。当然，如果你有大于10块GPU，那么下面代码会顺利执行。

```{.python .input  n=25}
import sys

try:
    nd.array([1,2,3], ctx=mx.gpu(0))
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```

我们可以通过`copyto`和`as_in_context`来在设备直接传输数据。

```{.python .input  n=26}
y = x.copyto(mx.gpu())
z = x.as_in_context(mx.gpu())
(y, z)
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "(\n [ 1.  2.  3.]\n <NDArray 3 @gpu(0)>, \n [ 1.  2.  3.]\n <NDArray 3 @gpu(0)>)"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

这两个函数的主要区别是，如果源和目标的context一致，`as_in_context`不复制，而`copyto`总是会新建内存：

```{.python .input  n=27}
yy = y.as_in_context(mx.gpu())
zz = z.copyto(mx.gpu())
(yy is y, zz is z)
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "(True, False)"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### GPU上的计算

计算会在数据的`context`上执行。所以为了使用GPU，我们只需要事先将数据放在上面就行了。结果会自动保存在对应的设备上：

```{.python .input  n=28}
nd.exp(z + 2) * y
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "\n[  20.08553696  109.19629669  445.23950195]\n<NDArray 3 @gpu(0)>"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

注意所有计算要求输入数据在同一个设备上。不一致的时候系统不进行自动复制。这个设计的目的是因为设备之间的数据交互通常比较昂贵，我们希望用户确切的知道数据放在哪里，而不是隐藏这个细节。下面代码尝试将CPU上`x`和GPU上的`y`做运算。

```{.python .input  n=29}
try:
    x + y
except mx.MXNetError as err:
    sys.stderr.write(str(err))
```

```{.json .output n=29}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "[10:35:49] src/imperative/./imperative_utils.h:55: Check failed: inputs[i]->ctx().dev_mask() == ctx.dev_mask() (2 vs. 1) Operator broadcast_add require all inputs live on the same context. But the first argument is on cpu(0) while the 2-th argument is on gpu(0)\n\nStack trace returned 10 entries:\n[bt] (0) /home/kegao/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2a9e78) [0x7efe43829e78]\n[bt] (1) /home/kegao/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2aa288) [0x7efe4382a288]\n[bt] (2) /home/kegao/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x24b308c) [0x7efe45a3308c]\n[bt] (3) /home/kegao/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x24c1bfe) [0x7efe45a41bfe]\n[bt] (4) /home/kegao/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(+0x2403a7b) [0x7efe45983a7b]\n[bt] (5) /home/kegao/.conda/envs/gluon/lib/python3.6/site-packages/mxnet/libmxnet.so(MXImperativeInvokeEx+0x63) [0x7efe45983fe3]\n[bt] (6) /home/kegao/.conda/envs/gluon/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so(ffi_call_unix64+0x4c) [0x7efeb040e5b0]\n[bt] (7) /home/kegao/.conda/envs/gluon/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so(ffi_call+0x1f5) [0x7efeb040dd55]\n[bt] (8) /home/kegao/.conda/envs/gluon/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so(_ctypes_callproc+0x3dc) [0x7efeb040589c]\n[bt] (9) /home/kegao/.conda/envs/gluon/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so(+0x9df3) [0x7efeb03fddf3]\n\n"
 }
]
```

### 默认会复制回CPU的操作

如果某个操作需要将NDArray里面的内容转出来，例如打印或变成numpy格式，如果需要的话系统都会自动将数据copy到主内存。

```{.python .input  n=30}
print(y)
print(y.asnumpy())
print(y.sum().asscalar())
```

```{.json .output n=30}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[ 1.  2.  3.]\n<NDArray 3 @gpu(0)>\n[ 1.  2.  3.]\n6.0\n"
 }
]
```

## Gluon的GPU计算

同NDArray类似，Gluon的大部分函数可以通过`ctx`指定设备。下面代码将模型参数初始化在GPU上：

```{.python .input  n=31}
from mxnet import gluon
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize(ctx=mx.gpu())
```

输入GPU上的数据，会在GPU上计算结果

```{.python .input  n=32}
data = nd.random.uniform(shape=[3,2], ctx=mx.gpu())
net(data)
```

```{.json .output n=32}
[
 {
  "data": {
   "text/plain": "\n[[ 0.02386322]\n [ 0.02697672]\n [ 0.0315817 ]]\n<NDArray 3x1 @gpu(0)>"
  },
  "execution_count": 32,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

确认下权重：

```{.python .input  n=33}
net[0].weight.data()
```

```{.json .output n=33}
[
 {
  "data": {
   "text/plain": "\n[[ 0.0301265   0.04819721]]\n<NDArray 1x2 @gpu(0)>"
  },
  "execution_count": 33,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 总结

通过`context`我们可以很容易在不同的设备上计算。

## 练习

- 试试大一点的计算任务，例如大矩阵的乘法，看看CPU和GPU的速度区别。如果是计算量很小的任务呢？
- 试试CPU和GPU之间传递数据的速度
- GPU上如何读写模型呢？

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/988)
