# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Pandas 入门
#
# Pandas  经常和其他数值计算工具一起使用, 并支持大部分  Numpy  风格的数组计算.
#
# Pandas  虽然采用了很多  Numpy  风格的代码, 但是  pandas  主要是用来处理异质型数据的, 而  Numpy  适合处理同质型数据.
#
# 以下为约定俗成的常用导入方式:

# %%
import pandas as pd
from pandas import Series, DataFrame

import numpy as np

# %% [markdown]
# ## Pandas  数据结构
#
# 学习  Pandas  需要熟悉两个常用的工具数据结构:  Series  和  DataFrame  .
# %% [markdown]
# ### Series
#
# Series  是一种一维的数组型对象, 它包含一个值序列, 并包含数据标签, 称为索引.
#
# 简单的序列可以仅仅由一个数组形成.

# %%
obj = pd.Series([4, 7, -5, 3])
obj

# %% [markdown]
# 如果不为数据指定索引, 那么默认生成的索引是从  0  到  N-1  .
#
# 可以通过  `values`  属性和  `index`  属性分别获得  Series  对象的值和索引:

# %%
obj.values


# %%
obj.index

# %% [markdown]
# 通常需要创建一个索引序列, 用标签去标识每个数据点:

# %%
obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
obj2

# %% [markdown]
# 这时, 你可以像字典一样, 使用标签去索引:

# %%
obj2["a"]


# %%
obj2["d"] = 6

# %% [markdown]
# 你也可以使用索引列表:

# %%
obj2[["c", "a", "d"]]

# %% [markdown]
# 同时也支持  Numpy  风格的数组操作. 同时这些操作会保留索引值的连接.

# %%
obj2[obj2 > 0]


# %%
obj2 * 2


# %%
np.exp(obj2)

# %% [markdown]
# 你也可以把  Series  当作字典类型使用, 因为本质上是非常相似的:

# %%
"b" in obj2


# %%
"e" in obj2

# %% [markdown]
# 如果你已经有了一个用字典存储的数据, 那就可以使用字典生成一个  Series  :

# %%
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3

# %% [markdown]
# 如果直接传送一个字典进去, 那么  Series  会自动根据字典序来排列键值对. 如果你不希望排序, 那么可以传入一个数组, 让其生成一个符合预期的  Series  .

# %%
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
obj4

# %% [markdown]
# 需要注意上面那个例子:
#
# 1. 其中只有  3  个值被正确放置了.
# 2. 因为字典中并没有  `California`  , 所以生成的  Series  中  `California`  对应的值是  `NaN`  .
# 3. 因为传入的序列中, 没有给出  `Utah`  这个键值, 所有这个值没有在  Series  中出现
# %% [markdown]
# 在  Pandas  中, 使用  `isnull`  和  `notnull`  来检查缺失的数据:

# %%
pd.isnull(obj4)


# %%
pd.notnull(obj4)

# %% [markdown]
# 你也可以在实例中直接调用这个两个函数:

# %%
obj4.isnull()

# %% [markdown]
# 自动对齐索引是  Series  中一个非常有用的特性:

# %%
obj3


# %%
obj4


# %%
obj3 + obj4

# %% [markdown]
# Series  对象和索引都有  `name`  属性:

# %%
obj4.name = "population"
obj4.index.name = "state"
obj4

# %% [markdown]
# Series  的索引可以通过一个列表, 按位置进行改变:

# %%
obj


# %%
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]
obj

# %% [markdown]
# ### DataFrame
#
# DataFrame  表示的是矩阵的数据表, 它包含已排序的列集合, 每一列可以是不同的值.
#
# DataFrame  既有行索引, 也有列索引. 数据被存储成一个以上的二维块.
# %% [markdown]
# 虽然  DataFrame  是二维的, 但可以利用分层索引展现更高维的数据.
# %% [markdown]
# 构建  DataFrame  最常见的方式是利用包含相等长度的列表或者  Numpy  数组的字典来形成  DataFrame  :

# %%
data = {
    "state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
    "year": [2000, 2001, 2002, 2001, 2002, 2003],
    "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2],
}
frame = pd.DataFrame(data)

# %% [markdown]
# DataFrame  会自动为  Series  分配索引.

# %%
frame

# %% [markdown]
# 使用  `head()`  方法可以直选出头部的  5  行:

# %%
frame.head()

# %% [markdown]
# 也可以使用列表指定列的顺序:

# %%
pd.DataFrame(data, columns=["year", "state", "pop"])

# %% [markdown]
# 如果传进去的列在字典中不存在, 那么会用缺失值来填充:

# %%
frame2 = pd.DataFrame(
    data,
    columns=["year", "state", "pop", "debt"],
    index=["one", "two", "three", "four", "five", "six"],
)
frame2

# %% [markdown]
# 你可以通过字典的方式或者属性来检索  Series  :

# %%
frame2["state"]


# %%
frame2.year

# %% [markdown]
# 返回的  Series  与原  DataFrame  有相同的索引, 且  Series  的  name  属性也会被合理地设置.
# %% [markdown]
# 行索引则可以通过位置或者属性  `loc`  进行选取:

# %%
frame2.loc["three"]

# %% [markdown]
# 列的值可以修改, 修改的方式可以参考  Numpy  数组的广播, 亦可以是标量, 亦可以是数组:

# %%
frame2["debt"] = 16.5
frame2


# %%
frame2.debt = np.arange(6.0)
frame2

# %% [markdown]
# 放你想把列表或者数组赋值给一个列时, 必须保证长度和  DataFrame  的长度相匹配.
# %% [markdown]
# 如果你将一个  Series  赋给一列, Series  索引将会按照  DataFrame  的索引重新排列, 并在空缺的地方填充缺失值:

# %%
val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])
frame2.debt = val
frame2

# %% [markdown]
# 如果被赋值的列并不存在, 则会生成一个新的列:

# %%
frame2["eastern"] = frame2.state == "Ohio"
frame2

# %% [markdown]
# 可以使用  del  方法移除列:

# %%
del frame2["eastern"]
frame2.columns

# %% [markdown]
# 对  DataFrame  的选取出来的  Series  是视图而不是拷贝, 对  Series  的修改会映射到  DataFrame  中.
#
# 如果需要拷贝, 需要显式地调用  `copy()`  方法.
# %% [markdown]
# 另一种常见的数据形式是嵌套字典:

# %%
pop = {"Nevada": {2001: 2.4, 2002: 2.9}, "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = pd.DataFrame(pop)
frame3

# %% [markdown]
# 可以使用类似  Numpy  的语法进行转置:

# %%
frame3.T

# %% [markdown]
# 与之前一样, 如果显式指明索引, 内部字典的键不会被排序:

# %%
pd.DataFrame(pop, index=[2001, 2002, 2003])

# %% [markdown]
# 也可使用包含  Series  的字典构造:

# %%
pdata = {"Ohio": frame3["Ohio"][:-1], "Nevada": frame3["Nevada"][:2]}
pd.DataFrame(pdata)

# %% [markdown]
# 可以向  DataFrame  构造函数传递的对象列表见下表:
# %% [markdown]
# 如果索引和列拥有  name  属性, 那么这些  name  属性也会被显示:

# %%
frame3.index.name = "year"
frame3.columns.name = "state"
frame3

# %% [markdown]
# 使用  values  属性可以将数据以二维的  ndarray  的形式返回:

# %%
frame3.values

# %% [markdown]
# 如果他们之间是不同的  dtypes  , 那么会自动选择最适合的  dtypes

# %%
frame2.values

# %% [markdown]
# ### 索引对象
#
# pandas  中的索引对象适用于存储轴标签和其他元数据的.
#
# 在构造  Series  或者  DataFrame  时, 你所以使用的任意数组或者序列都可以在内部转换为索引对象:

# %%
obj = pd.Series(range(3), index=["a", "b", "c"])
index = obj.index
index


# %%
index[1:]

# %% [markdown]
# 索引对象是不可表的, 用户无法修改索引对象:

# %%
# index[1] = 'd'

# %% [markdown]
# 不变性是的在多种数据结构中分享索引对象更安全:

# %%
labels = pd.Index(np.arange(3))
labels


# %%
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2


# %%
obj2.index is labels

# %% [markdown]
# 除了类似数组, 索引对象也像一个固定大小的集合:

# %%
frame3


# %%
frame3.columns


# %%
"Ohio" in frame3.columns


# %%
2003 in frame3.index

# %% [markdown]
# 与  Python  的集合不同, pandas  对象可以包含重复标签:

# %%
dup_labels = pd.Index(["foo", "foo", "bar", "bar"])
dup_labels

# %% [markdown]
# 索引对象一些常用的属性和方法表格:
# %% [markdown]
# ## 基本功能
#
# %% [markdown]
# Series  和  DataFrame  中数据交互的基础机制。
# %% [markdown]
# ### 重建索引
#
# `reindex`  是  pandas  对象的重要方法, 用于创建一个符合新索引的新对象:

# %%
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
obj

# %% [markdown]
# Series  调用  `reindex`  方法时, 会将数据按照新的索引进行排列.
#
# 如果某个索引值之前并不存在, 则会引入缺失值来占位:

# %%
obj2 = obj.reindex(["a", "b", "c", "d", "e"])
obj2

# %% [markdown]
# 在重建索引时可能需要插值或者填值, `method`  可选参数允许我们使用诸如  `ffill`  等方法在重建索引时插值, `ffill`  方法会将值向前填充.

# %%
obj3 = pd.Series(["bule", "purple", "yellow"], index=[0, 2, 4])
obj3


# %%
obj3.reindex(range(6), method="ffill")

# %% [markdown]
# 在  DataFrame  中, `reindex`  可以改变行索引, 列索引, 也可以同时改变二者.
#
# 当只传入一个序列时, 结果中的行会重建索引:

# %%
frame = pd.DataFrame(
    np.arange(9).reshape((3, 3)),
    index=["a", "c", "d"],
    columns=["Ohio", "Texas", "California"],
)
frame


# %%
frame2 = frame.reindex(["a", "b", "c", "d"])
frame2

# %% [markdown]
# 使用  `columns`  可以重建列索引:

# %%
states = ["Texas", "Utah", "California"]
frame.reindex(columns=states)

# %% [markdown]
# 可以使用  loc  进行更简洁的标签索引:

# %%
frame.loc[["a", "b", "c", "d"], states]

# %% [markdown]
# `reindex`  方法的参数：
# %% [markdown]
# ### 按轴删除条目
#
# `drop`  方法可以返回对应的标签删除后的新结果:

# %%
obj = pd.Series(np.arange(5.0), index=["a", "b", "c", "d", "e"])
obj


# %%
new_obj = obj.drop("c")
new_obj


# %%
obj.drop(["d", "c"])

# %% [markdown]
# 在  DataFrame  中, 索引值可以根据轴来删除.

# %%
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=["Ohio", "Colorado", "Utah", "New York"],
    columns=["one", "two", "three", "four"],
)
data

# %% [markdown]
# 默认情况下不指定时, `drop`  根据行标签删除条目:

# %%
data.drop(["Colorado", "Ohio"])

# %% [markdown]
# 同时可以通过  `axis=1`  或者  `axis=columns`  来删除列值:

# %%
data.drop("two", axis=1)


# %%
data.drop(["two", "four"], axis="columns")

# %% [markdown]
# 当指定了参数  inplace  之后, `drop`  就会改变原对象, 不返回新对象.

# %%
obj.drop("c", inplace=True)
obj

# %% [markdown]
# ### 索引、选择与过滤
#
# Series  的索引与  Numpy  数组索引的功能类似, 只不过  Series  的索引值可以不是整数.

# %%
obj = pd.Series(np.arange(4.0), index=["a", "b", "c", "d"])
obj


# %%
obj["b"]


# %%
obj[1]


# %%
obj[2:4]


# %%
obj[["b", "a", "d"]]


# %%
obj[[1, 3]]


# %%
obj[obj < 2]

# %% [markdown]
# Python  中的切片不含尾部, 但  Series  的切片不同:

# %%
obj["b":"c"]

# %% [markdown]
# 如果切片没有使用整数值, 那么切片结果是包含尾部的
# %% [markdown]
# Series  同样支持切片赋值:

# %%
obj["b":"c"] = 5
obj

# %% [markdown]
# DataFrame  的索引也类似:

# %%
data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=["Ohio", "Colorabo", "Utah", "New York"],
    columns=["one", "two", "three", "four"],
)
data


# %%
data["two"]


# %%
data[["three", "one"]]

# %% [markdown]
# 行选择语法与布尔表达式:

# %%
data[:2]


# %%
data[data["three"] > 5]

# %% [markdown]
# 布尔表达式索引:

# %%
data < 5


# %%
data[data < 5] = 0
data

# %% [markdown]
# #### 使用 `loc` 和 `iloc` 选择数据
#
# `loc`  轴标签, `iloc`  参数标签
#
# 可以利用这两个特殊索引符号以  Numpy  的风格去除数组的行列子集:

# %%
# data
data.loc["Colorabo", ["two", "three"]]


# %%
data.iloc[2, [3, 0, 1]]


# %%
data.iloc[2]


# %%
data.iloc[[1, 2], [3, 0, 1]]

# %% [markdown]
# 同时也可以用于切片:

# %%
data.loc[:"Utah", "two"]


# %%
data.iloc[:, :3][data.three > 5]

# %% [markdown]
# DataFrame  索引选项表:
# %% [markdown]
# ### 整数索引
#
# pandas  对象在使用整数索引的时候对新用户来说经常会产生歧义, 因为与  Python  内建的数据结构索引有所不同.
#
# 比如:

# %%
ser = pd.Series(np.arange(3.0))
ser

# %% [markdown]
# 此时索引  -1  , pandas  不知道你到底是像索引  -1  还是  '-1'  ,  然后报错
# %% [markdown]
# 假如使用非整数索引, 就不会有潜在的歧义:

# %%
# ser[-1]
ser2 = pd.Series(np.arange(3.0), index=["a", "b", "c"])
ser2[-1]

# %% [markdown]
# 基于这种情况, 如果你的数据包含整数的轴索引, 请始终使用标签索引.
# %% [markdown]
# 或者使用更精确的  `loc`  和  `iloc`  区分两种索引:

# %%
ser[:1]


# %%
ser.loc[:1]


# %%
ser.iloc[:1]

# %% [markdown]
# ### 算术和数据对齐
#
# 当你将对象相加时, 如果存在某个索引对不相同时, 则返回结果的索引将是索引对的并集.

# %%
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=["a", "c", "d", "e"])
s2 = pd.Series([2.1, 3.6, -1.5, 4, 3.1], index=["a", "c", "e", "f", "g"])


# %%
s1


# %%
s2


# %%
s1 + s2

# %% [markdown]
# 在没有重叠的标签位置上, 运算导致了其值为缺失值.
#
# 在  DataFrame  上, 行和列都会执行类似的对齐:

# %%
df1 = pd.DataFrame(
    np.arange(9.0).reshape((3, 3)),
    columns=list("bcd"),
    index=["Ohio", "Texas", "Colorado"],
)
df2 = pd.DataFrame(
    np.arange(12.0).reshape((4, 3)),
    columns=list("bde"),
    index=["Utah", "Ohio", "Texas", "Oregon"],
)


# %%
df1


# %%
df2


# %%
df1 + df2

# %% [markdown]
# 如果行列完全不相同, 那么结果将全部为空:

# %%
df1 = pd.DataFrame({"A": [1, 2]})
df2 = pd.DataFrame({"B": [3, 4]})


# %%
df1


# %%
df2


# %%
df1 - df2

# %% [markdown]
# #### 使用填充值的算术方法
#
# 应对当产生缺失值时怎么替换缺失值, 变成需要的值.

# %%
df1 = pd.DataFrame(np.arange(12.0).reshape((3, 4)), columns=list("abcd"))
df2 = pd.DataFrame(np.arange(20.0).reshape((4, 5)), columns=list("abcde"))
df2.loc[1, "b"] = np.nan


# %%
df1


# %%
df2


# %%
df1 + df2

# %% [markdown]
# 此时在  `fill_value`  参数上, 提供出现缺失值时的值:

# %%
df1.add(df2, fill_value=0)

# %% [markdown]
# 类似算数方法之类前面带  r  前缀的都是翻转版的函数:

# %%
1 / df1


# %%
df1.rdiv(1)

# %% [markdown]
# 重建索引也可能产生控制, 这是也可以利用这个参数自动指定一个值:

# %%
df1.reindex(columns=df2.columns, fill_value=0)

# %% [markdown]
# 算术方法表:
# %% [markdown]
# #### DataFrame  和  Series  之间的操作
#
# 类似  Numpy  广播风格的算术运算:

# %%
arr = np.arange(12.0).reshape((3, 4))


# %%
arr


# %%
arr[0]


# %%
arr - arr[0]

# %% [markdown]
# 在  DataFrame  中这个也是类似的:

# %%
frame = pd.DataFrame(
    np.arange(12.0).reshape((4, 3)),
    columns=list("bde"),
    index=["Utah", "Ohio", "Texas", "Oregon"],
)
series = frame.iloc[0]


# %%
frame


# %%
series


# %%
frame - series

# %% [markdown]
# 如果索引值不在  DataFrame  的列中, 也不再  Series  的索引中, 那么会重建索引:

# %%
series2 = pd.Series(range(3), index=["b", "e", "f"])
frame + series2

# %% [markdown]
# 如果想要改变广播方向, 在列上进行广播, 在行上匹配, 就必须使用算术方法的其中一种:

# %%
series3 = frame["d"]


# %%
frame


# %%
series3


# %%
frame.sub(series3, axis="index")

# %% [markdown]
# ### 函数应用和映射
# %% [markdown]
# Numpy  的通用函数也可以用在  Pandas  上:

# %%
frame = pd.DataFrame(
    np.random.randn(4, 3),
    columns=list("bde"),
    index=["Utah", "Ohio", "Texas", "Oregon"],
)
frame


# %%
np.abs(frame)

# %% [markdown]
# 另外一个常用功能是, 将函数应用到一行或者一列的一维数组上, 可以使用  `apply`  方法来实现这个功能:

# %%
f = lambda x: x.max() - x.min()
frame.apply(f)


# %%
frame.apply(f, axis="columns")

# %% [markdown]
# 大部分时候都是用内置的方法, 基本不需要自定义一个  `apply`
# %% [markdown]
# 该函数也不一定是个标量, 也可以是  Series

# %%
def f(x):
    return pd.Series([x.min(), x.max()], index=["min", "max"])


frame.apply(f)

# %% [markdown]
# 调用  `applymap`  方法可以实现逐元素执行的方法：

# %%
format = lambda x: "%.2f" % x
frame.applymap(format)

# %% [markdown]
# Series  有一个  `map`  方法, 功能是一致的

# %%
frame["e"].map(format)

# %% [markdown]
# ### 排序和排名
#
# 使用  `sort_index`  可以按需对行或者列索引进行字典序排序.
#
# 该方法返回一个新的, 排序好的对象:

# %%
obj = pd.Series(range(4), index=["d", "a", "b", "c"])
obj.sort_index()


# %%
frame = pd.DataFrame(
    np.arange(8).reshape((2, 4)), index=["three", "one"], columns=["d", "a", "b", "c"]
)
frame.sort_index()


# %%
frame.sort_index(axis=1)

# %% [markdown]
# 同时也支持降序排序

# %%
frame.sort_index(axis=1, ascending=False)

# %% [markdown]
# 如果希望根据值来排序, 那么需要使用  `sort_value`  方法:

# %%
obj = pd.Series([4, 7, -3, 2])
obj.sort_values()

# %% [markdown]
# 默认情况下缺失值会被排序到最后:

# %%
obj = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()

# %% [markdown]
# 对  DataFrame  排序时, 可以使用一列或者多列作为排序键:

# %%
frame = pd.DataFrame({"b": [4, 7, -3, -2], "a": [0, 1, 0, 1]})
frame


# %%
frame.sort_values(by="b")


# %%
frame.sort_values(by=["a", "b"])

# %% [markdown]
# `rank`  函数可以获取排名, 默认情况下, 当出现相同的标签是会用平均排名:

# %%
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()

# %% [markdown]
# 下面例子就是不使用平均排名的方式:

# %%
obj.rank(method="first")

# %% [markdown]
# 或者还可以降序排名:

# %%
obj.rank(ascending=False, method="max")

# %% [markdown]
# 改变平级关系的表:
# %% [markdown]
# DataFrame  也可以计算排名:

# %%
frame = pd.DataFrame({"b": [4.3, 7, -3, 2], "a": [0, 1, 0, 1], "c": [-2, 5, 8, -2.5]})
frame

# %% [markdown]
# ### 含有重复标签的轴索引
#
# 虽然某些索引函数要求标签唯一, 但这并不是强制性的.

# %%
obj = pd.Series(range(5), index=["a", "a", "b", "b", "c"])
obj


# %%
obj.index.is_unique

# %% [markdown]
# 此时你去索引它, 他就会返回多条数据:

# %%
obj["a"]


# %%
obj["c"]

# %% [markdown]
# 在  DataFrame  中这个情况也是一样的

# %%
df = pd.DataFrame(np.random.randn(4, 3), index=["a", "a", "b", "b"])
df


# %%
df.loc["b"]

# %% [markdown]
# ## 描述性统计的概述与计算
#
# 与  Numpy  内建的类似的方法相比, Pandas  增加了对缺失值的处理.

# %%
df = pd.DataFrame(
    [[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
    index=["a", "b", "c", "d"],
    columns=["one", "two"],
)
df

# %% [markdown]
# 默认情况下, 返回的是列的加和情况, 并自动排除  NaN  值

# %%
df.sum()

# %% [markdown]
# 指定轴后可以实现行加和.

# %%
df.sum(axis=1)

# %% [markdown]
# 如果不希望排除缺失值, 可以指定  `skipna`  参数.

# %%
df.mean(axis="columns", skipna=False)

# %% [markdown]
# 规约方法的可选参数:
# %% [markdown]
# 一些方法返回的是间接统计信息:

# %%
df.idxmax()

# %% [markdown]
# 除了规约方法, 还有积累型方法:

# %%
df.cumsum()

# %% [markdown]
# 也有既不是积累型也不是规约型的方法.
#
# `describe`  可以一次性产生多个汇总统计.

# %%
df.describe()

# %% [markdown]
# 如果是非数值型数据, `describe`  会产生另一种汇总统计:

# %%
obj = pd.Series(["a", "a", "b", "c"] * 4)
obj.describe()

# %% [markdown]
# 相关方法的完成列表:
# %% [markdown]
# ### 相关性和协方差
#
#

# %%
price = pd.read_pickle("example/yahoo_price.pkl")
volume = pd.read_pickle("example/yahoo_volume.pkl")

# %% [markdown]
# 计算股价百分比:

# %%
returns = price.pct_change()
returns.tail()

# %% [markdown]
# `corr`  计算的是相关性, 两个  Series  中重叠的, 非  NA  的, 按索引对齐的值.
#
# 对应的, `cov`  计算的是协方差.

# %%
returns["MSFT"].corr(returns["IBM"])


# %%
returns["MSFT"].cov(returns["IBM"])

# %% [markdown]
# 用更简洁的语法完成任务:

# %%
returns.MSFT.corr(returns.IBM)

# %% [markdown]
# 如果使用的是  DataFrame  返回的则是相关性和协方差矩阵.

# %%
returns.corr()


# %%
returns.cov()

# %% [markdown]
# `corrwith`  可以计算某一行或者列与其他序列的相关性.

# %%
returns.corrwith(returns.IBM)


# %%
returns.corrwith(volume)

# %% [markdown]
# ### 唯一值 计数和成员属性
#

# %%
obj = pd.Series(["c", "a", "d", "a", "a", "b", "b", "c", "c"])


# %%
uniques = obj.unique()
uniques

# %% [markdown]
# 唯一值不是按顺序来进行的, 如果需要排序, 可以使用  `uniques.sort()`  .
#

# %%
obj.value_counts()

# %% [markdown]
# 顶层的  `value_counts`  也可以使用, 也用于任意数组或者序列:

# %%
pd.value_counts(obj.values, sort=False)

# %% [markdown]
# `isin`  可以用作成员属性检查, 还可以将数据集以  Series  或  DataFrame  一列的形式过滤子集.

# %%
obj


# %%
mask = obj.isin(["b", "c"])
mask


# %%
obj[mask]

# %% [markdown]
# `Index.get_indexer`  可以将一个数组转换为另一个唯一值数组.

# %%
to_match = pd.Series(["c", "a", "b", "b", "c", "a"])
unique_vals = pd.Series(["c", "b", "a"])
pd.Index(unique_vals).get_indexer(to_match)

# %% [markdown]
# 唯一值 计数和集合成员属性方法:
# %% [markdown]
# 直方图统计:

# %%
data = pd.DataFrame(
    {"Qu1": [1, 3, 4, 3, 4,], "Qu2": [2, 3, 1, 2, 3,], "Qu3": [1, 5, 2, 4, 4,],}
)
data


# %%
result = data.apply(pd.value_counts).fillna(0)
result


# %%

