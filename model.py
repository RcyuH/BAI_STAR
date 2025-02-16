#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:31:03 2025

@author: rcyuh
"""

"""
******Steps******

B1: Tính semantic similiraty
    +) Input: df chứa infor của items gồm title, description, 
           category, brand, sales ranking, price.
    +) Prompt vào LLM để nó cho ra embedding vector tương ứng
    +) Tính cosine similirity cho đôi một item embedding vector 
    +) Output: Ma trận Rs (size NxN, N là số items trong system)

B2: Tính collaborative commonality
    +) Input: User-Item matrix C, input đặc trưng của bài toán đề xuất
              Giá trị trong đó là ratings, number of clicks,... (Chưa thấy đoạn nào đề cập đến cái lày)
    +) Tính cosine similiratity cho đôi Stepsmột "hàng trong C"  
    +) Output: Ma trận Rc (size NxN, N là số items trong system)

B3: Tính score tổng hợp từ hai chỉ số trên cho các unseen item 
(NHỚ LÀ CHỈ CHO CÁC UNSEEN ITEMS NHA)
    +) Công thức (1) trong paper
    +) Với mỗi user u, ta sẽ cần tính score cho tất cả unseen items của u, tức các items i thuộc (I - Su)
    +) Output: Với mỗi user u => Top k unseen items có highest score 
        => Có thể lưu vào 1 matrix (ndarray) với size MxK (M là số users trong system)

B4: Ranking 
    +) Cấu trúc của 1 prompt: Figure 3 in 
    +) Chú ý 2 dạng pair-wise và list-wise nên sử dụng giải thuật window slide để tăng tốc độ thuật toán

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
path = "/home/rcyuh/Desktop/BAI/Quá trình học/Bước 3/data/"

# meta_data_df = getDF("/home/rcyuh/Desktop/BAI/Quá trình học/Bước 3/data/Beauty_Metadata.json.gz")
meta_data_df = getDF(path + 'Beauty_Metadata.json.gz')
review_df = getDF(path + 'Beauty_Review.json.gz')
user_item_df = pd.read_csv(path + "Beauty_User-Item_Matrix.csv", names=["itemID", "userID", "rating", "timestamp"], header=None)

























