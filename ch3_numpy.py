
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   numpy.py
@Time    :   2021/06/25 09:41:43
@Author  :   hyong 
@Version :   1.0
@Contact :   hyong_cs@outlook.com
'''
# here put the import lib

# %%

import numpy as np

# * 创建一个列表
data_1 = [1, 2, 3, 4]

# * 数组化
ndarray_1 = np.array(data_1)
type(ndarray_1)

# ! output: numpy.ndarray

# %%

# * 使用元组传进去
tuple_1 = (1, 2, 3, 4)
ndarray_1 = np.array(tuple_1)
type(ndarray_1)

# ! numpy.ndarray

# %%

# * zeros  函数的功能是创建  n  个  0  元素构成的数组
np.zeros(10)
# ! output: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

# %%

print(np.zeros((2, 2)))
# ! output: [[0. 0.]
# !          [0. 0.]]

# %%

data_2 = np.arange(1, 10, 1)
data_2

# ! output: array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# %%

np.linspace(0, 1, 10)
# ! output: array([0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
# !                0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ])

# %%

print(np.logspace(0, 2, 5))
# ! output: [  1.           3.16227766  10.          31.6227766  100.        ]

# %%

# * 创建一个二维数组
arr = np.arange(12).reshape((3, 4))
print(arr)
# ! output: [[ 0  1  2  3]
# !          [ 4  5  6  7]
# !          [ 8  9 10 11]]

# %%

# * 拆回一维数组
print(arr.ravel())
# ! output: [ 0  1  2  3  4  5  6  7  8  9 10 11]

# %%

# * 可以指定顺序
print(arr.ravel('F'))
# ! output: [ 0  4  8  1  5  9  2  6 10  3  7 11]

# %%

soccer = np.array(['C罗', '梅西', '本泽马'])
# * 数组的索引从  0  开始
soccer[0]
# ! output: 'C罗'

# %%

# * 左闭右开，无法取到  2  对应的值
soccer[0:2]
# ! output: array(['C罗', '梅西'], dtype='<U3')

# %%

test_soccer = soccer[0]
test_soccer = '阿扎尔'
soccer
# ! output: array(['C罗', '梅西', '本泽马'], dtype='<U3')

# %%

# * 创建二维数组
city = np.array([['北京', '上海', '广州', '深圳'],
                 ['天津', '长沙', '杭州', '武汉'],
                 ['重庆', '成都']])
city
# ! output: array([list(['北京', '上海', '广州', '深圳']), list(['天津', '长沙', '杭州', '武汉']),
# !                list(['重庆', '成都'])], dtype=object)

# %%

# * 对齐的二维数组
city = np.array([['北京', '上海',],
                 ['天津', '长沙',],
                 ['重庆', '成都',]])
city

# ! output: array([['北京', '上海'],
# !                ['天津', '长沙'],
# !                ['重庆', '成都']], dtype='<U2')

# %%

city[0]
# ! output: array(['北京', '上海'], dtype='<U2')

# %%

city[0] = '北京'
city
# ! output: array([['北京', '北京'],
# !                ['天津', '长沙'],
# !                ['重庆', '成都']], dtype='<U2')

# %%

# * 第一种索引方式
city[0][1]
# ! output: '北京'

# %%

# * 第二种索引方式
city[2,1]
# ! output: '成都'

# %%

# * 切片索引
city[0:2, 0:1]
# ! output: array([['北京'],
# !                ['天津']], dtype='<U2')

# %%

city_3d = np.array([[['北京', '上海', '广州'],
                     ['深圳', '杭州', '苏州']],
                    [['青岛', '济南', '合肥'],
                     ['合肥', '西安', '徐州']]])
city_3d
# ! output: array([[['北京', '上海', '广州'],
# !                 ['深圳', '杭州', '苏州']],
# !                [['青岛', '济南', '合肥'],
# !                 ['合肥', '西安', '徐州']]], dtype='<U2')

# %%

city_3d[0]
# ! output: array([['北京', '上海', '广州'],
# !                ['深圳', '杭州', '苏州']], dtype='<U2')

# %%

city_3d[0] = '北京'
city_3d
# ! output: array([[['北京', '北京', '北京'],
# !                 ['北京', '北京', '北京']],
# !                [['青岛', '济南', '合肥'],
# !                 ['合肥', '西安', '徐州']]], dtype='<U2')

# %%

champion = np.array(['FNC', 'TPA', 'SKT1', 'SSW', 'SKT1', 'SKT1', 'SSW'])
data = np.random.randn(7, 5)
data
# ! output: array([[ 0.1484297 , -1.46068075, -0.49992988, -0.46977362,  0.65299666],
# !                [-0.65700028,  0.39633905, -0.56698742,  0.51575065, -0.6302347 ],
# !                [ 0.90491563, -0.77444038, -1.78202984, -0.15461069, -0.62365176],
# !                [-1.02129805,  1.02843319, -0.67581709,  1.02320674,  0.74636294],
# !                [ 0.09546156,  1.44338694,  1.02768775,  0.76028571, -1.98660206],
# !                [-0.49720203,  1.20116477, -1.99240124,  0.25941805,  0.66190578],
# !                [ 1.73374047,  1.57164426, -0.7733701 , -0.97211007,  0.57917163]])

# %%

# * 当等式结果为  True  时，索引结果
data[champion == 'SKT1']
# ! output: array([[ 0.90491563, -0.77444038, -1.78202984, -0.15461069, -0.62365176],
# !                [ 0.09546156,  1.44338694,  1.02768775,  0.76028571, -1.98660206],
# !                [-0.49720203,  1.20116477, -1.99240124,  0.25941805,  0.66190578]])

# %%

# * 切片操作
data[champion == 'SKT1', :3]
# ! output: array([[ 0.90491563, -0.77444038, -1.78202984],
# !                [ 0.09546156,  1.44338694,  1.02768775],
# !                [-0.49720203,  1.20116477, -1.99240124]])

# %%

# * 支持逻辑表达式
data[(champion == 'SKT1') | (champion != 'SSW'), :3]
# ! output: array([[ 0.1484297 , -1.46068075, -0.49992988],
# !                [-0.65700028,  0.39633905, -0.56698742],
# !                [ 0.90491563, -0.77444038, -1.78202984],
# !                [ 0.09546156,  1.44338694,  1.02768775],
# !                [-0.49720203,  1.20116477, -1.99240124]])

# %%

# * 阈值过滤
data[data < 0] = 0
data
# ! output: array([[0.1484297 , 0.        , 0.        , 0.        , 0.65299666],
# !                [0.        , 0.39633905, 0.        , 0.51575065, 0.        ],
# !                [0.90491563, 0.        , 0.        , 0.        , 0.        ],
# !                [0.        , 1.02843319, 0.        , 1.02320674, 0.74636294],
# !                [0.09546156, 1.44338694, 1.02768775, 0.76028571, 0.        ],
# !                [0.        , 1.20116477, 0.        , 0.25941805, 0.66190578],
# !                [1.73374047, 1.57164426, 0.        , 0.        , 0.57917163]])

# %%

num = np.arange(20).reshape(4, 5)
num
# ! output :array([[ 0,  1,  2,  3,  4],
# !                [ 5,  6,  7,  8,  9],
# !                [10, 11, 12, 13, 14],
# !                [15, 16, 17, 18, 19]])

# %%

# * 花式索引会  copy  出新数组
# * 同时使用整数数组进行索引
num[[0, 1, 2, 3], [0, 1, 2, 3]]
# ! output: array([ 0,  6, 12, 18])

# %%

boradcast = np.array([1, 2, 3, 4, 5])
scalar = 1
boradcast - scalar
# ! output: array([0, 1, 2, 3, 4])

# %%

boradcast_2d = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10]])
boradcast_2d - boradcast
# ! output: array([[0, 0, 0, 0, 0],
# !                [5, 5, 5, 5, 5]])

# %%

boradcast_3d = np.array([[[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10]],
                         [[11, 12, 13, 14, 15],
                          [16, 17, 18, 19, 20]]])
boradcast_3d - scalar
# ! output: array([[[ 0,  1,  2,  3,  4],
# !                 [ 5,  6,  7,  8,  9]],
# !                [[10, 11, 12, 13, 14],
# !                 [15, 16, 17, 18, 19]]])

# %%

boradcast_3d - boradcast_2d

# ! output: array([[[ 0,  0,  0,  0,  0],
# !                 [ 0,  0,  0,  0,  0]],
# !                [[10, 10, 10, 10, 10],
# !                 [10, 10, 10, 10, 10]]])

# %%

# * 一维数组求和
cal = np.array([1, 2, 3, 4, 5,])
sum(cal)
# ! output: 15

# %%

# * 二维数组求和
cal_2d = np.array([[1, 2, 3, 4, 5,], [6, 7, 8, 9, 10]])
sum(cal_2d)
# ! output: array([ 7,  9, 11, 13, 15])

# %%

# * 二维数组求和指定轴
# * 必须使用  np.sum()  来求和，不能使用自带  np.sum()
np.sum(cal_2d, axis=1)
# ! output: array([15, 40])

# %%

# * 加权平均的计算算法
grade = np.array([92, 98, 90, 70,])
credit = np.array([4, 5, 2, 10,])
GPA = np.sum(grade * credit) / np.sum(credit)
GPA

# ! output: 82.76190476190476

# %%

# * 使用自带的加权平均
np.average(grade, weights=credit)
# ! output: 82.76190476190476

# %%

# * 算最小值
arr_1 = np.array([2, 1, 3])
min(arr_1)
# ! output: 1

# %%

# * sort 排序函数
arr_1.sort()
arr_1
# ! output: array([1, 2, 3])

# %%

# * 使用 np.sort 不会改变原数组
arr_1 = np.array([2, 1, 3])
np.sort(arr_1)
arr_1
# ! output: array([2, 1, 3])

# %%

# * 点积运算
a = np.array([[1, 2,], [3, 4,]])
a * a
# ! output: array([[ 1,  4],
# !                [ 9, 16]])

# %%

# * 矩阵乘积
np.dot(a, a)
# ! output: array([[ 7, 10],
# !                [15, 22]])


# %%

# * matrix 类 求逆
b = np.mat(a)
b.I
# ! output: matrix([[-2. ,  1. ],
# !                 [ 1.5, -0.5]])

# %%

# * matrix 类 矩阵乘积
b * b
# ! output: matrix([[ 7, 10],
# !                 [15, 22]])

# %%

# * matrix  类  求点积
np.multiply(b, b)
# ! output: matrix([[ 1,  4],
# !                 [ 9, 16]])

# %%

# * reduce 求和
arr_2 = np.array([[1, 2, 3, 4, 5,], [2, 3, 4, 5, 6,]])
np.add.reduce(arr_2)
# ! output: array([ 3,  5,  7,  9, 11])

# %%

# * 累计求和
arr_3 = np.array([[1, 2, 3, 4, 5,], [2, 3, 4, 5, 6,], [3, 4, 5, 6, 7,]])
np.add.accumulate(arr_3)
# ! output: array([[ 1,  2,  3,  4,  5],
# !                [ 3,  5,  7,  9, 11],
# !                [ 6,  9, 12, 15, 18]])

# %%

# * 自定义  ufunc  函数
b = np.mat(np.array([[1, 2, ], [3, 4, ]]))
def customed_multiply(x, y):
    return x * y
multiply_them = np.frompyfunc(customed_multiply, 2, 1)
multiply_them(b, b)
# ! output: matrix([[1, 4],
# !                 [9, 16]], dtype=object)

# %%

# * 测试两者之间的速度
import time
import math

x = [i * 0.001 for i in range(1000000)]
start = time.time()
for i, t in enumerate(x):
    x[i] = math.sin(t)
print(f'math.sin: {time.time() - start}')

t = [i * 0.001 for i in range(1000000)]
t = np.array(t)
start = time.time()
np.sin(t)
print(f"numpy.sin:{time.time() - start}")
# ! output: math.sin: 0.38187551498413086
# !         numpy.sin:0.030726194381713867

# %%

# * 创建多项式
a = np.array([1, -2, -3, ])
p = np.poly1d(a)
p(0)
# ! output: -3

# %%

p
# ! output: poly1d([ 1, -2, -3])

# %%

p * p
# ! output: poly1d([ 1, -4, -2, 12,  9])

# %%

p + [2, 1]
# ! output: poly1d([ 1,  0, -2])

# %%

p + 22
# ! output: poly1d([ 1, -2, 19])

# %%

np.roots(p)
# ! output: array([ 3., -1.])

# %%

# * unique 函数
arr_4 = np.array([1, 1, 2, 2, 3, 3, 4, 4, ])
np.unique(arr_4, return_index=True, return_inverse=True)
# ! output: (array([1, 2, 3, 4]), array([0, 2, 4, 6]), array([0, 0, 1, 1, 2, 2, 3, 3]))

# %%

arr_5 = np.array([1, 2, 3, 4, 5, 5, 6, 7, 9, 9, ])
np.bincount(arr_5)
# ! output: array([0, 1, 1, 1, 1, 2, 1, 1, 0, 2])

# %%

np.histogram(arr_5, bins=5, range=(0, 9))
# ! output: (array([1, 2, 3, 2, 2]), array([0. , 1.8, 3.6, 5.4, 7.2, 9. ]))

# %%



# %%
