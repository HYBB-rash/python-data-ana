
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ch4_pands.py
@Time    :   2021/07/02 23:30:48
@Author  :   hyong 
@Version :   1.0
@Contact :   hyong_cs@outlook.com
'''
# here put the import lib

# %% [markdown]

"""

## Series 的创建

Series 类似于一维数组对象，区别是带了索引

"""
# %%

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# %%

s = Series([17, 17.81, 17.90, 18, 19, np.nan])

# %%

s
# ! output: 0    17.00
# !         1    17.81
# !         2    17.90
# !         3    18.00
# !         4    19.00
# !         5      NaN
# !         dtype: float64

# %%
