# 使用NDArray来处理数据

对于机器学习来说，处理数据往往是万事之开头。它包含两个部分：(i)数据读取，(ii)数据已经在内存中时如何处理。本章将关注后者。

我们首先介绍`NDArray`，这是MXNet储存和变换数据的主要工具。如果你之前用过`NumPy`，你会发现`NDArray`和`NumPy`的多维数组非常类似。当然，`NDArray`提供更多的功能，首先是CPU和GPU的异步计算，其次是自动求导。这两点使得`NDArray`能更好地支持机器学习。

## 让我们开始

我们先介绍最基本的功能。如果你不懂我们用到的数学操作也不用担心，例如按元素加法、正态分布；我们会在之后的章节分别从数学和代码编写的角度详细介绍。

我们首先从`mxnet`导入`ndarray`这个包

```{.python .input  n=1}
from mxnet import ndarray as nd
```

然后我们创建一个3行和4列的2D数组（通常也叫**矩阵**），并且把每个元素初始化成0

```{.python .input  n=2}
nd.zeros((3, 4))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  0.  0.  0.]\n [ 0.  0.  0.  0.]\n [ 0.  0.  0.  0.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

类似的，我们可以创建数组每个元素被初始化成1。

```{.python .input  n=3}
x = nd.ones((3, 4))
x
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

或者从python的数组直接构造

```{.python .input  n=4}
nd.array([[1,2],[2,3]])
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  2.]\n [ 2.  3.]]\n<NDArray 2x2 @cpu(0)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们经常需要创建随机数组，即每个元素的值都是随机采样而来，这个经常被用来初始化模型参数。以下代码创建数组，它的元素服从均值0标准差1的正态分布。

```{.python .input  n=7}
y = nd.random_normal(0, 1, shape=(3, 4))
y
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "\n[[-0.7882176   0.74177277 -1.47344387 -1.07309282]\n [-1.04248273 -1.32788491 -1.47496617 -0.52414197]\n [ 1.26625562  0.89506418 -0.60159451  1.20405591]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

跟`NumPy`一样，每个数组的形状可以通过`.shape`来获取

```{.python .input  n=8}
y.shape
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "(3, 4)"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

它的大小，就是总元素个数，是形状的累乘。

```{.python .input  n=9}
y.size
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "12"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=10}
x
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 操作符

NDArray支持大量的数学操作符，例如按元素加法：

```{.python .input  n=11}
y
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "\n[[-0.7882176   0.74177277 -1.47344387 -1.07309282]\n [-1.04248273 -1.32788491 -1.47496617 -0.52414197]\n [ 1.26625562  0.89506418 -0.60159451  1.20405591]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=12}
x + y
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "\n[[ 0.2117824   1.74177277 -0.47344387 -0.07309282]\n [-0.04248273 -0.32788491 -0.47496617  0.47585803]\n [ 2.26625562  1.89506412  0.39840549  2.20405579]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=16}
x[0,2]=5
x
```

```{.json .output n=16}
[
 {
  "data": {
   "text/plain": "\n[[ 1.  1.  5.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 16,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

乘法：

```{.python .input  n=17}
x * y
```

```{.json .output n=17}
[
 {
  "data": {
   "text/plain": "\n[[-0.7882176   0.74177277 -7.36721945 -1.07309282]\n [-1.04248273 -1.32788491 -1.47496617 -0.52414197]\n [ 1.26625562  0.89506418 -0.60159451  1.20405591]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 17,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

指数运算：

```{.python .input  n=18}
nd.exp(y)
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "\n[[ 0.45465446  2.09965444  0.22913502  0.34194928]\n [ 0.35257822  0.26503724  0.22878647  0.59206313]\n [ 3.54754424  2.44749284  0.54793727  3.3336103 ]]\n<NDArray 3x4 @cpu(0)>"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

也可以转置一个矩阵然后计算矩阵乘法：

```{.python .input  n=19}
nd.dot(x, y.T)
```

```{.json .output n=19}
[
 {
  "data": {
   "text/plain": "\n[[ -8.48675728 -10.26934052   0.3574031 ]\n [ -2.59298134  -4.36947536   2.76378107]\n [ -2.59298134  -4.36947536   2.76378107]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 19,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=20}
nd.dot(y,x.T)
```

```{.json .output n=20}
[
 {
  "data": {
   "text/plain": "\n[[ -8.48675728  -2.59298134  -2.59298134]\n [-10.26934052  -4.36947536  -4.36947536]\n [  0.3574031    2.76378107   2.76378107]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 20,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

我们会在之后的线性代数一章讲解这些运算符。

## 广播（Broadcasting）

当二元操作符左右两边ndarray形状不一样时，系统会尝试将其复制到一个共同的形状。例如`a`的第0维是3, `b`的第0维是1，那么`a+b`时会将`b`沿着第0维复制3遍：

```{.python .input  n=23}
nd.arange(3)
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "\n[ 0.  1.  2.]\n<NDArray 3 @cpu(0)>"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=24}
a = nd.arange(3).reshape((3,1))
b = nd.arange(2).reshape((1,2))
print('a:', a)
print('b:', b)
print('a+b:', a+b)

```

```{.json .output n=24}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "a: \n[[ 0.]\n [ 1.]\n [ 2.]]\n<NDArray 3x1 @cpu(0)>\nb: \n[[ 0.  1.]]\n<NDArray 1x2 @cpu(0)>\na+b: \n[[ 0.  1.]\n [ 1.  2.]\n [ 2.  3.]]\n<NDArray 3x2 @cpu(0)>\n"
 }
]
```

## 跟NumPy的转换

ndarray可以很方便地同numpy进行转换

```{.python .input  n=25}
import numpy as np
x = np.ones((2,3))
y = nd.array(x)  # numpy -> mxnet
z = y.asnumpy()  # mxnet -> numpy
print([z, y])
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[array([[ 1.,  1.,  1.],\n       [ 1.,  1.,  1.]], dtype=float32), \n[[ 1.  1.  1.]\n [ 1.  1.  1.]]\n<NDArray 2x3 @cpu(0)>]\n"
 }
]
```

```{.python .input  n=26}
type(z)
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "numpy.ndarray"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=28}
type(x)
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "numpy.ndarray"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 替换操作

在前面的样例中，我们为每个操作新开内存来存储它的结果。例如，如果我们写`y = x + y`, 我们会把`y`从现在指向的实例转到新建的实例上去。我们可以用Python的`id()`函数来看这个是怎么执行的：

```{.python .input  n=32}
x = nd.ones((3, 4))
y = nd.ones((3, 4))

before = id(y)
y = y + x
id(y) == before
print(x)
print(y)
print(before)
print(id(y))
```

```{.json .output n=32}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]\n [ 1.  1.  1.  1.]]\n<NDArray 3x4 @cpu(0)>\n\n[[ 2.  2.  2.  2.]\n [ 2.  2.  2.  2.]\n [ 2.  2.  2.  2.]]\n<NDArray 3x4 @cpu(0)>\n140461003155440\n140461033016064\n"
 }
]
```

我们可以把结果通过`[:]`写到一个之前开好的数组里：

```{.python .input  n=36}
z = nd.zeros_like(x)
print(z)
before = id(z)
z[:] = x + y
id(z) == before
```

```{.json .output n=36}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[ 0.  0.  0.  0.]\n [ 0.  0.  0.  0.]\n [ 0.  0.  0.  0.]]\n<NDArray 3x4 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 36,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

但是这里我们还是为`x+y`创建了临时空间，然后再复制到`z`。需要避免这个开销，我们可以使用操作符的全名版本中的`out`参数：

```{.python .input  n=37}
nd.elemwise_add(x, y, out=z)
id(z) == before
```

```{.json .output n=37}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 37,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

如果现有的数组不会复用，我们也可以用 `x[:] = x + y` ，或者 `x += y` 达到这个目的：

```{.python .input  n=40}
before = id(x)
x[:] =x+ y
id(x) == before
```

```{.json .output n=40}
[
 {
  "data": {
   "text/plain": "True"
  },
  "execution_count": 40,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 截取（Slicing）

MXNet NDArray 提供了各种截取方法。截取 x 的 index 为 1、2 的行：

```{.python .input  n=42}
x = nd.arange(0,9).reshape((3,3))
print('x: ', x)
x[0:2]
```

```{.json .output n=42}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x:  \n[[ 0.  1.  2.]\n [ 3.  4.  5.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 0.  1.  2.]\n [ 3.  4.  5.]]\n<NDArray 2x3 @cpu(0)>"
  },
  "execution_count": 42,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

以及直接写入指定位置：

```{.python .input  n=43}
x[1,2] = 9.0
x
```

```{.json .output n=43}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  1.  2.]\n [ 3.  4.  9.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

多维截取：

```{.python .input  n=47}
x = nd.arange(0,9).reshape((3,3))
print('x: ', x)
x[1:2,1:3]
```

```{.json .output n=47}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "x:  \n[[ 0.  1.  2.]\n [ 3.  4.  5.]\n [ 6.  7.  8.]]\n<NDArray 3x3 @cpu(0)>\n"
 },
 {
  "data": {
   "text/plain": "\n[[ 4.  5.]]\n<NDArray 1x2 @cpu(0)>"
  },
  "execution_count": 47,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

多维写入：

```{.python .input  n=49}
x[1:3,1:3] = 9.0
x
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "\n[[ 0.  1.  2.]\n [ 3.  9.  9.]\n [ 6.  9.  9.]]\n<NDArray 3x3 @cpu(0)>"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## 总结

ndarray模块提供一系列多维数组操作函数。所有函数列表可以参见[NDArray API文档](https://mxnet.incubator.apache.org/api/python/ndarray.html)。

**吐槽和讨论欢迎点**[这里](https://discuss.gluon.ai/t/topic/745)
