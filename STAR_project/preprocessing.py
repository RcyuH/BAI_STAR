#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:53:44 2025

@author: rcyuh
"""

import pandas as pd
import gzip
import json
from item_embedding import ItemEmbeddingGenerator

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

class preProcessing_metadata:
    def __init__(self):
        pass
    
    def extract_6_core(self, items_df: pd.DataFrame) -> pd.DataFrame
        # Trích xuất 6 thuộc tính chính: title, description, category, brand, sales ranking, price
        pass
    
    def drop_duplicate(self):
        # Xóa các product có id trùng và giữ lại cái đầu tiên
        pass
    
    def price_processing(self):
        
    
    def rank_processing(self):
    
    def convert_df_to_dict(self, items_df: pd.DataFrame) -> dict  
        items_dict = items_df.set_index("asin").to_dict(orient="index")
        
        return items_dict
    
    def main_flow(self):
        pass
    
meta_data_df_cleaned = meta_data_df.drop_duplicates(subset=["asin"], keep="first")    

pre = preProcessing()
meta_data_dict = pre.convert_df_to_dict(meta_data_df_cleaned)

generator = ItemEmbeddingGenerator()
generator.debug_prompt(meta_data_dict)
    