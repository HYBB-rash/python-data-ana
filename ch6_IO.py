
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ch6_IO.py
@Time    :   2021/07/13 10:20:49
@Author  :   hyong 
@Version :   1.0
@Contact :   hyong_cs@outlook.com
'''
# here put the import lib

# %%

import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# %% [markdown]

"""

## 文本格式数据的读写

"""

# %%

!cat example/ex1.csv

# %%

df = pd.read_csv('example/ex1.csv')
df

# %%

df = pd.read_table('example/ex1.csv', sep=',')
df

# %%

!cat example/ex2.csv

# %%

pd.read_csv('example/ex2.csv', header=None)

# %%

pd.read_csv('example/ex2.csv',
            names=['a', 'b', 'c', 'd', 'message'])

# %%

names = ['a', 'b', 'c', 'd', 'message']
pd.read_csv('example/ex2.csv', 
            names=names,
            index_col='message')

# %%

!cat example/csv_mindex.csv

# %%

parsed = pd.read_csv('example/csv_mindex.csv',
                     index_col=['key1', 'key2'])
parsed

# %%

list(open('example/ex3.txt'))

# %%

result = pd.read_table('example/ex3.txt', sep='\s+')
result

# %%

!cat example/ex4.csv

# %%

pd.read_csv('example/ex4.csv',
            skiprows=[0, 2, 3])

# %%

!cat example/ex5.csv

# %%

result = pd.read_csv('example/ex5.csv')
result

# %%

pd.isnull(result)

# %%

result = pd.read_csv('example/ex5.csv',
                     na_values=['NULL'])
result

# %%

sentinels = {'message': ['foo', 'NA'],
             'something': ['two']}
pd.read_csv('example/ex5.csv', na_values=sentinels)

# %% [markdown]

"""

### 分块读入文本文件

"""

# %%

pd.options.display.max_rows = 10
result = pd.read_csv('example/ex6.csv')
result

# %%

pd.read_csv('example/ex6.csv', nrows=5)

# %%

chunker = pd.read_csv('example/ex6.csv', chunksize=1000)
chunker

# %%

chunker = pd.read_csv('example/ex6.csv', chunksize=1000)

tot = pd.Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)

tot[:10]

# %% [markdown]

"""

### 将数据写入文本格式

"""

# %%

data = pd.read_csv('example/ex5.csv')
data

# %%

data.to_csv('example/out.csv')
!cat example/out.csv

# %%

import sys
data.to_csv(sys.stdout, sep='|')

# %%

data.to_csv(sys.stdout, na_rep='NULL')

# %%

data.to_csv(sys.stdout, index=False, header=False)

# %%

data.to_csv(sys.stdout, index=False, 
            columns=['a', 'b', 'c'])

# %%

dates = pd.date_range('1/1/2000', periods=7)
ts = pd.Series(np.arange(7), index=dates)
ts.to_csv('example/tseries.csv')
!cat example/tseries.csv 

# %% [markdown]

"""

### 使用分割格式

"""

# %%

!cat example/ex7.csv

# %%

import csv
f = open('example/ex7.csv')
reader = csv.reader(f)

# %%

for line in reader:
    print(line)

# %%

with open('example/ex7.csv') as f:
    lines = list(csv.reader(f))

# %%

header, values = lines[0], lines[1:]

# %%

data_dict = {h: v for h, v in zip(header, zip(*values))}
data_dict

# %%

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL

f = open('example/ex7.csv')
reader = csv.reader(f, dialect=my_dialect)

# %%

reader = csv.reader(f, delimiter='|')

# %%

with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))

# %% [markdown]

"""

### JSON  数据

"""

# %%

obj = """
{
    "name": "Wes",
    "places_lived": ["United States", "Spain", "Germany"],
    "pet": null,
    "siblings": [
        {"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
        {"name": "Katie", "age": 38, "pets": ["Sixes", "Stache", "Cisco"]}
    ]
}
"""

# %%

import json


# %%

result = json.loads(obj)
result

# %%

asjson = json.dumps(result)
asjson

# %%

siblings = pd.DataFrame(result['siblings'], 
                        columns=['name', 'age'])
siblings

# %%

!cat example/example.json

# %%

data = pd.read_json('example/example.json')
data

# %%

print(data.to_json())

# %%

print(data.to_json(orient='records'))

# %% [markdown]

"""

### XML, HTML, 网络抓取

"""
# %%

tables = pd.read_html(
    'example/fdic_failed_bank_list.html'
)
len(tables)

# %%

failures = tables[0]
failures.head()

# %%

close_timestamps = pd.to_datetime(
    failures['Closing Date']
)
close_timestamps.dt.year.value_counts()

# %% [markdown]

"""

#### 使用  lxml.objectfy  解析  XML

"""

# %%

from lxml import objectify

path = 'datasets/mta_perf/Performance_MNR.xml'
parsed = objectify.parse(open(path))
root = parsed.getroot()

# %%

data = []
skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ',
               'DESIRED_CHANGE', 'DECIMAL_PLACES']

for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)

# %%

perf = pd.DataFrame(data)
perf.head()

# %%

from io import StringIO
tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()

# %%

root

# %%

root.get('href')

# %%

root.text

# %% [markdown]

"""

## 二进制格式

"""

# %%

frame = pd.read_csv('example/ex1.csv')
frame

# %%

frame.to_pickle('example/frame_pickle')

# %% [markdown]

"""

### 使用  HDF5  格式

"""


# %%

frame = pd.DataFrame(
    {'a': np.random.randn(1000)}
)

store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
store

# %%

store['obj1']

# %%

store.put('obj2', frame, format='table')
store.select('obj2',
             where=['index >= 10 and index <= 15'])

# %%

store.close()

# %%

frame.to_hdf('mydata.h5', 'obj3', 
             format='table')
pd.read_hdf('mydata.h5', 'obj3',
            where=['index < 5'])

# %% [markdown]

"""

### 读取  Microsoft Excel  文件

"""

# %%

xlsx = pd.ExcelFile('example/ex1.xlsx')

# %%

pd.read_excel(xlsx, 'Sheet1')

# %%

frame = pd.read_excel('example/ex1.xlsx', 'Sheet1')
frame

# %%

writer = pd.ExcelWriter('example/ex2.xlsx')
frame.to_excel(writer, 'Sheet1')
writer.save()

# %%

frame.to_excel('example/ex2.xlsx')

# %% [markdown]

"""

## 与  web api  交互

"""

# %%

import requests

url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(url)

resp

# %%

data = resp.json()
data[0]['title']

# %%

issues = pd.DataFrame(data, columns=['number', 'title',
                                     'labels', 'state'])
issues

# %% [markdown]

"""

## 与数据库交互

"""

# %%

import sqlite3

query = """
CREATE TABLE test (
    a VARCHAR(20), b VARCHAR(20),
    c REAL       , d INTEGER
)
"""

con = sqlite3.connect('mydata.sqlite')

# %%

con.execute(query)
con.commit()

# %%

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES (?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()

# %%

cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows

# %%

cursor.description

# %%

pd.DataFrame(rows, 
             columns=[x[0] for x in cursor.description])

# %%

import sqlalchemy as sqla

db = sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test', db)

# %% [markdown]

"""

## end

"""

# %%

!rm mydata.csv mydata.h5 mydata.sqlite

# %%
