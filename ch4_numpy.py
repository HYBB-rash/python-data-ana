# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Numpy 科学计算库
#
# ndarry  支持类似标量语法的运算，且  numpy  在大型数学运算上与原生  Python  相比速度快且占用内存少。以下是测试代码与测试结果。速度在  10  倍到  100  倍之间。

# %%
import numpy as np

my_arr = np.arange(1000000)
my_list = list(range(1000000))


# %%
get_ipython().run_line_magic("time", "for _ in range(10): my_arr2 = my_arr * 2")


# %%
get_ipython().run_line_magic(
    "time", "for _ in range(10): my_list2 = [x * 2 for x in my_list]"
)

# %% [markdown]
# ## Numpy ndarry：多维数组对象
#
# ndaray  支持类似标量语法的运算。
#
# 先随机生成数据：

# %%
import numpy as np

data = np.random.randn(2, 3)
data

# %% [markdown]
# 然后对  data  进行一些数学操作：

# %%
data * 10


# %%
data + data

# %% [markdown]
# ndarry  是一个多维同类的数据容器，每个元素都是相同元素
#
# 每个数组都有  shape  ，表示数组在对应维度上的数量
#
# 每个数组都有  dtype  ，用于描述数组的数据类型

# %%
data.shape


# %%
data.dtype

# %% [markdown]
# ## 生成  ndarry
#
# 可以使用列表生成：

# %%
data1 = [
    6,
    7.5,
    8,
    0,
    1,
]
arr1 = np.array(data1)
arr1

# %% [markdown]
# 也可以使用嵌套序列：

# %%
data2 = [[1, 2, 3, 4,], [5, 6, 7, 8,]]
arr2 = np.array(data2)
arr2

# %% [markdown]
# 想确定数组的维数，可以使用  ndim  属性确定数组的维度：

# %%
arr2.ndim

# %% [markdown]
# shape  属性返回一个  tuple  ，可以确定得到每一维的长度

# %%
arr2.shape

# %% [markdown]
# 除非显式指定，否则  np.array  会自动推断出数组的数据类型，存储在  dtype  中

# %%
arr1.dtype


# %%
arr2.dtype

# %% [markdown]
# 如果需要创建高维的数组，就需要为  shape  传递一个数组。
# %% [markdown]
# zeros  函数可以创造全零数组：

# %%
np.zeros(10)


# %%
np.zeros((3, 6))

# %% [markdown]
# ones  可以创造全一数组。
# %% [markdown]
# empty  可以创建一个没有初始化的数组。
#
# 不要用  empty  生成全零数组，它可能会存在非零值。

# %%
np.empty((2, 3, 2))

# %% [markdown]
# arange  是  python  的内置函数  range  的数组版：

# %%
np.arange(15)

# %% [markdown]
# 常用的函数表，numpy  在没有注明的情况下，默认生成  float64  类型的数据:
# %% [markdown]
# ## ndarry 数据类型
#
# 数据类型  dtype  是  ndarry  为某一种类型数据所声明的内存块信息，也称为元数据。

# %%
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3,], dtype=np.int32)


# %%
arr1.dtype


# %%
arr2.dtype

# %% [markdown]
# 常用数据类型表：
# %% [markdown]
# 可以使用  astype  的方法显式转换数组的数据类型

# %%
arr = np.array([1, 2, 3, 4, 5])


# %%
arr.dtype


# %%
float_arr = arr.astype(np.float64)


# %%
float_arr.dtype

# %% [markdown]
# 如果是小数转整数，那么小数部分会被去掉

# %%
arr = np.array([3, 7, -1.2, -2.6, 0.5, 12.9, 10.1,])


# %%
arr


# %%
arr.astype(np.int32)

# %% [markdown]
# 如果数据是字符串，那么可以使用  astype  将其转换为数字

# %%
numeric_strings = np.array(["1.25", "-9.6", "42",], dtype=np.string_)
numeric_strings.astype(float)

# %% [markdown]
# 如果因为某些情况无法转换，将会抛出一个  ValueError
# %% [markdown]
# 你可以使用另一个数组的  dtype  属性：

# %%
int_array = np.arange(10)
clibers = np.array([0.22, 0.270, 0.357, 0.380, 0.44, 0.50,], dtype=np.float64)
int_array.astype(clibers.dtype)

# %% [markdown]
# 也可以使用代码指定类型

# %%
empty_uint32 = np.empty(8, dtype="u4")
empty_uint32

# %% [markdown]
# **注意**，使用  astype  总是会生成一个新的数组，即使数据是一样的
# %% [markdown]
# ## 数组运算
#
# 向量化，可以让你不需要循环操作就完成大量的运算工作，在不同规模的数组中使用算数运算叫做广播

# %%
arr = np.array([[1.0, 2.0, 3.0,], [4.0, 5.0, 6.0,]])
arr

# %% [markdown]
# 直接数组规模的乘法运算：

# %%
arr * arr


# %%
arr - arr

# %% [markdown]
# 也可以直接与标量计算：

# %%
1 / arr


# %%
arr ** 0.5

# %% [markdown]
# 数组之间可以进行比较，生成一个  bool  数组：

# %%
arr2 = np.array([[0.0, 4.0, 1.0,], [7.0, 2.0, 12.0,]])
arr2


# %%
arr2 > arr

# %% [markdown]
# ## 索引
#
# 在一维数组时，与  python  自带的索引切片用法相同

# %%
arr = np.arange(10)
arr


# %%
arr[5]


# %%
arr[5:8]

# %% [markdown]
# 而对切片赋值会对所有对应位置都附上值：

# %%
arr[5:8] = 12
arr

# %% [markdown]
# numpy  的切片视图不是副本，而是直接修改原数组，切片返回的就是原数组而不是副本，需要注意
#
# 为了速度，为了内存，  numpy  的数组切片不会复制副本

# %%
arr_slice = arr[5:8]
arr_slice

# %% [markdown]
# 当改变切片返回的数组，原数组也会改变：

# %%
arr_slice[1] = 12345
arr


# %%
arr_slice[:] = 64
arr

# %% [markdown]
# 如果希望返回副本，那么需要显式的指定一下才可以，例如:

# %%
arr[5:8].copy()

# %% [markdown]
# 二维数组索引返回的是一个数组：

# %%
arr_2d = np.array([[1, 2, 3,], [4, 5, 6,], [7, 8, 9,]])
arr_2d[2]

# %% [markdown]
# 如果需要准确的索引，可以有两种方式完成
#
# 第一个是索引行，第二个是索引列

# %%
arr_2d[0][2]


# %%
arr_2d[0, 2]

# %% [markdown]
# 上升到更高维的也同理，返回的是低一维度的结果：

# %%
arr_3d = np.array([[[1, 2, 3,], [4, 5, 6,]], [[7, 8, 9,], [10, 11, 12,]]])
arr_3d

# %% [markdown]
# 高纬度的索引也同理：

# %%
arr_3d[0]
old_values = arr_3d[0].copy()
arr_3d[0] = 42
arr_3d

# %% [markdown]
# 可以直接赋值：

# %%
arr_3d[0] = old_values
arr_3d

# %% [markdown]
# 可以加逗号索引

# %%
arr_3d[1, 0]


# %%
x = arr_3d[1]
x


# %%
x[0]

# %% [markdown]
# 用切片索引时，一维的时候与  python  自带的索引相同：

# %%
arr


# %%
arr[1:6]

# %% [markdown]
# 二维的时候就有点不同了，切片此时是在索引列：

# %%
arr_2d


# %%
arr_2d[:2]

# %% [markdown]
# 也可以两个维度同时切片索引：

# %%
arr_2d[:2, 1:]

# %% [markdown]
# 也可以一个切片索引，一个数字索引：

# %%
arr_2d[1, :2]


# %%
arr_2d[:2, 2]

# %% [markdown]
# 单独索引一列的技巧：

# %%
arr_2d[:, :1]

# %% [markdown]
# 也可以对切片索引赋值：

# %%
arr_2d[:2, 1:] = 0
arr_2d

# %% [markdown]
# ## 布尔索引
#
# 先随机生成一部分数据：

# %%
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.random.randn(7, 4)
names


# %%
data

# %% [markdown]
# names  数组和字符串  'Bob'  能产生一个布尔数组：

# %%
names == "Bob"

# %% [markdown]
# 索引数组的时候可以使用布尔数组：

# %%
data[names == "Bob"]

# %% [markdown]
# 当布尔数组长度不正确的时候，布尔值选择数据的方法并不会报错，需要小心这个地方。
# %% [markdown]
# 同时还可以和其他索引方式一起使用：

# %%
data[names == "Bob", 2:]


# %%
data[names == "Bob", 3]

# %% [markdown]
# 同时也支持其他布尔表达式的使用，比如  `!=`  ,  `~`

# %%
names != "Bob"


# %%
data[~(names == "Bob")]

# %% [markdown]
# `~`  是取反。你可以对在任意的数组前使用这个表示取反。

# %%
cond = names == "Bob"
data[~cond]

# %% [markdown]
# 当简单的布尔表达式不能满足需求时，复杂的表的式同样也可以使用：

# %%
mask = (names == "Bob") | (names == "Will")
mask


# %%
data[mask]

# %% [markdown]
# Python  关键字  and  和  or  在布尔数组中不起作用，需要用  &  和  |  来代替。
# %% [markdown]
# 同时你可以利用布尔数组做索引，然后对对应的位置赋值：

# %%
data[data < 0] = 0
data

# %% [markdown]
# 你也可以分别对每一行或者每一列设置数值：

# %%
data[names != "Joe"] = 7
data

# %% [markdown]
# ## 神奇索引
#
# 神奇索引使用整数数组来索引数据。
# %% [markdown]
# 生成一个  8 * 4  的数组：

# %%
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr

# %% [markdown]
# 你可以通过传递一个数组，索引指定顺序：

# %%
arr[[4, 3, 0, 6]]

# %% [markdown]
# 同时你负数索引也没有问题：

# %%
arr[[-3, -5, -7]]

# %% [markdown]
# 也可以传入多个数组索引，但此时就是根据每个索引的元素顺序，指定一个坐标，最终结果是生成一个一维数组：

# %%
arr = np.arange(32).reshape((8, 4))
arr


# %%
arr[[1, 5, 7, 2,], [0, 3, 1, 2,]]

# %% [markdown]
# 神奇索引的结果总是一维的。
# %% [markdown]
# 神奇索引总是把数据复制到一个新的数组中。

# %%
arr[[1, 5, 7, 2,]][:, [0, 3, 1, 2,]]

# %% [markdown]
# ## 数组的转置和换轴
#
# 使用  T  属性可以将数组转置：

# %%
arr = np.arange(15).reshape((3, 5))
arr


# %%
arr.T

# %% [markdown]
# 可以使用  `np.dot()`  计算矩阵内积：

# %%
arr = np.random.randn(6, 3)
arr


# %%
np.dot(arr.T, arr)
# np.dot(arr, arr.T)

# %% [markdown]
# 而使用  `transpose()`  可以制定置换轴，再转置：

# %%
arr = np.arange(16).reshape((2, 2, 4))
arr


# %%
arr.transpose((1, 0, 2))

# %% [markdown]
# 使用  `swapaxes()`  可以对数组里的轴进行陶正，重组数据：

# %%
arr


# %%
arr.swapaxes(1, 2)

# %% [markdown]
# ## 通用函数，快速逐元素数组函数
#
# 通用函数，又称  `ufunc`  ，是一种在  ndarray  数据中进行逐元素操作的函数。
#
# 通用函数就是对一些简单函数的向量化封装
# %% [markdown]
# 很多  `ufunc`  是简单的逐元素转换，比如  `sqrt`  或  `exp`  函数：

# %%
arr = np.arange(10)
arr


# %%
np.sqrt(arr)


# %%
np.exp(arr)

# %% [markdown]
# 有些通用函数比如  `add`  或  `maximum`  则会接收两个数组并返回一个数组作为结果,  因此称为二元通用函数:

# %%
x = np.random.randn(8)
y = np.random.randn(8)
x


# %%
y


# %%
np.maximum(x, y)

# %% [markdown]
# 这里的  `maximum`  将两个向量较大的元素筛选了出来.
# %% [markdown]
# 也用通用函数返回多个数组,比如  `modf`  ,  是  python  内建函数  `divmod`  的向量化版本.
#
# 它会返回一个整数部分的数组和小数部分的数组:

# %%
arr = np.random.randn(7) * 5
arr


# %%
remainder, whole_part = np.modf(arr)
remainder


# %%
whole_part

# %% [markdown]
# 通用函数会提供一个参数  `out`  ,  你可以指定返回结果的接收位置:

# %%
arr


# %%
np.sqrt(arr)


# %%
np.sqrt(arr, arr)


# %%
arr

# %% [markdown]
# 还有一元通用函数表:
# %% [markdown]
# ## 使用数组进行面向数组编程
#
# 使用  Numpy  数组表达式能够完成多种数据操作任务, 无需写大量的循环, 称为向量化.
#
# 以下是简单示例:

# %%
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
ys


# %%
z = np.sqrt(xs ** 2 + ys ** 2)
z


# %%
from matplotlib import pyplot as plt

plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.title("Image plot $\sqrt{x^2, y^2}$ for a grid of values")

# %% [markdown]
# ### 将条件逻辑作为数组操作
#
# `numpy.where`  函数是三元表达式  `x if condition else y`  的向量化版本.
#

# %%
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])

# %% [markdown]
# 假设我们需要根据  cond  的情况选择  xarr  和  yarr  的元素, 那么使用列表推导式的代码如下:

# %%
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result

# %% [markdown]
# 如果使用  python  解释器完成这一工作, 那么会比较慢. 而使用  `numpy.where`  可以规避这一问题:

# %%
result = np.where(cond, xarr, yarr)
result

# %% [markdown]
# `np.where`  的后两个参数不一定要是向量, 可以是两个标量.
# %% [markdown]
# 假如你有一个随机生成的矩阵数据, 并且想将其中的正值替换成  2  , 所有的负值替换成  -2  :

# %%
arr = np.random.randn(4, 4)
arr


# %%
arr > 0


# %%
np.where(arr > 0, 2, -2)

# %% [markdown]
# 你也可以将向量和标量结合起来:

# %%
np.where(arr > 0, 2, arr)

# %% [markdown]
# ### 数学和统计方法
#
# Numpy  中, 可以使用聚合函数，或者叫做缩减函数。可以在实例调用，也可以在顶层调用
#

# %%
arr = np.random.randn(5, 4)
arr


# %%
arr.mean()


# %%
np.mean(arr)


# %%
arr.sum()

# %% [markdown]
# 像  `mean`, `sum`  等函数可以接收一个可选参数  `axis`  .
#
# axis = 1  计算的是每一列，  axis = 0  计算的是每一行

# %%
arr.mean(axis=1)


# %%
arr.sum(axis=0)

# %% [markdown]
# 像  `cumsum`  和  `cumprod`  的不会聚合，他们会产生中间结果

# %%
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr.cumsum()


# %%
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr

# %% [markdown]
# 这种函数也可以指定  `axis`  来确定聚合方向
#

# %%
arr.cumsum(axis=0)


# %%
arr.cumprod(axis=1)

# %% [markdown]
# 统计方法列表:
# %% [markdown]
# ### 布尔值数组的方法
#
# 布尔值是强制值  1  或者  0，因此  sum  可以用于计算布数组中  True  的个数

# %%
arr = np.random.randn(100)
(arr > 0).sum()

# %% [markdown]
# `any`  方法检查数组是否存在有一个  True  , `all`  检查是否每个值都是  True

# %%
bools = np.array([False, False, True, False])
bools.any()


# %%
bools.all()

# %% [markdown]
# ### 排序
#
# numpy  数组可以使用  sort  方法排序, 结果会在原数组上反映出来.

# %%
arr = np.random.randn(6)
arr


# %%
arr.sort()
arr

# %% [markdown]
# 二维数组可以指定排序的方向:

# %%
arr = np.random.randn(5, 3)
arr


# %%
arr.sort(1)
arr

# %% [markdown]
# 顶层的  sort  函数返回的是一个副本，而不是原数组

# %%
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]

# %% [markdown]
# ### 唯一值与其他集合逻辑
#
# Numpy  中一些关于集合的基本函数操作
#
# `np.unique`  返回数组里的所有唯一值，并排序:

# %%
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
np.unique(names)


# %%
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)

# %% [markdown]
# 下面是纯  python  实现:

# %%
sorted(set(names))

# %% [markdown]
# `np.in1d`  检查一个数组的值是否在另一个数组中，并返回一个布尔值数组

# %%
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

# %% [markdown]
# 集合函数列表:
# %% [markdown]
# ## 使用数组进行文件输入和输出
#
# `np.save`  和  `np.load`  是两大高效存储硬盘数据的工具函数.
#
# 数组在默认情况下是以压缩格式进行存储的, 后缀名是 .npy :

# %%
arr = np.arange(10)
np.save("some_array", arr)
np.load("some_array.npy")

# %% [markdown]
# 如果忘记写  .npy  了, 会自动帮忙加上.
# %% [markdown]
# 可以使用  `np.savez`  将数组作为参数传递给该函数, 可以用来保存多个数组:
# %% [markdown]
# 当载入的时候, 就能获得字典型的对象, 在字典用可以方便的载入单个数组:

# %%
np.savez("array_archive.npz", a=arr, b=arr)


# %%
arch = np.load("array_archive.npz")
arch["b"]

# %% [markdown]
# 如果你想往数据文件里再加入一个数组,  可以使用  `arrays_compressed.npz`  来将数据压入已经压缩的文件:

# %%
np.savez_compressed("arrays_compressed.npz", a=arr, b=arr)


# %%
get_ipython().system("rm array_archive.npz")
get_ipython().system("rm arrays_compressed.npz")
get_ipython().system("rm some_array.npy")

# %% [markdown]
# ## 线性代数
#
# numpy  中用于进行矩阵运算的操作
#
# 比如矩阵点乘, 可以使用  `dot`  函数:

# %%
x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = np.array([[6.0, 23.0], [-1, 7], [8, 9]])
x


# %%
y


# %%
x.dot(y)


# %%
np.dot(x, y)

# %% [markdown]
# 顶层的  `dot()`  函数和实例的是等价的.
# %% [markdown]
# 一个二维数组和一个长度合适的一维数组的点乘是一个一维数组, 不用去特定思考方向.

# %%
np.dot(x, np.ones(3))

# %% [markdown]
# 特殊符号  `@`  可以作为点乘的符号来使用:

# %%
x @ np.ones(3)

# %% [markdown]
# `numpy.linalg`  拥有一个矩阵分解的标准函数集, 以及其他常用函数, 例如求逆和行列式求解.

# %%
from numpy.linalg import inv, qr

X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)


# %%
mat.dot(inv(mat))


# %%
q, r = qr(mat)
r

# %% [markdown]
# 常用的  `numpy.linalg`  函数表:
#
# %% [markdown]
# ## 伪随机数生成
#
# `numpy.random`  可以高效生成多种概率分布下的完整样本值数组.
#
# 例如你可以用  `normal`  获取正态分布的数组:

# %%
samples = np.random.normal(size=(4, 4))
samples

# %% [markdown]
# `numpy.random`  生成随机数的速度快了一个数量级

# %%
from random import normalvariate

N = 1000000
get_ipython().run_line_magic(
    "timeit", "samples = [normalvariate(0, 1) for _ in range(N)]"
)


# %%
get_ipython().run_line_magic("timeit", "np.random.normal(size=N)")

# %% [markdown]
# 这些都是伪随机数, 可以通过  `np.random.seed`  改变  Numpy  的随机数种子

# %%
np.random.seed(1234)

# %% [markdown]
# 但上面的方法是全局的, 如果希望使用局部的, 可以使用  `np.random.RandomState`  创建一个随机数生成器.

# %%
rng = np.random.RandomState(1234)


# %%
rng.randn(10)

# %% [markdown]
# 随机函数表:
# %% [markdown]
# ## 随机漫步
#
# 随机漫步的一个示例

# %%
import random

position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
plt.plot(walk[:100])


# %%
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
walk.min()


# %%
walk.max()


# %%
(np.abs(walk) >= 10).argmax()


# %%
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(1)
walks


# %%
walks.max()


# %%
walks.min()


# %%
hist30 = (np.abs(walks) >= 30).any(1)
hist30


# %%
crossing_times = (np.abs(walks[hist30]) >= 30).argmax()
crossing_times.mean()


# %%
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))

