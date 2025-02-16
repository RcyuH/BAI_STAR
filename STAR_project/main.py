#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:31:03 2025

@author: rcyuh
"""

import pandas as pd
import gzip
import json

def parse(path):
    g = open(path, 'r', encoding="utf-8")
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
      df[i] = d
      i += 1
    return pd.DataFrame.from_dict(df, orient='index')

#CONST
path = "/home/rcyuh/Desktop/2. BAI/Quá trình học/Bước 3/data/"

meta_data_df = getDF(path + 'Beauty_Metadata.json.gz')
review_df = getDF(path + 'Beauty_Review.json.gz')
user_item_df = pd.read_csv(path + "Beauty_User-Item_Matrix.csv", names=["itemID", "userID", "rating", "timestamp"], header=None)

print(meta_data_df.isnull().sum())





























