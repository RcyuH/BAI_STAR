#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:53:44 2025

@author: rcyuh
"""

import pandas as pd
import numpy as np
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
    
    def extract_7_core(self, items_df: pd.DataFrame) -> pd.DataFrame:
        # Trích xuất 7 thuộc tính chính: title, description, category, brand, sales ranking, price, asin
        
        return items_df[['title', 'description', 'category', 'brand', 'rank', 'price', 'asin']] 
    
    def drop_duplicate(self, items_df: pd.DataFrame) -> pd.DataFrame:
        # Xóa các product có id trùng và giữ lại cái đầu tiên
        
        return items_df.drop_duplicates(subset=["asin"], keep="first")
    
    def price_processing(self, items_df: pd.DataFrame) -> pd.DataFrame:
        items_df['price'] = np.where(items_df['price'].astype(str).str.startswith("$"), items_df['price'], np.nan)
        
        return items_df
    
    def rank_processing(self,  items_df: pd.DataFrame) -> pd.DataFrame:  
        items_df["rank"] = items_df['rank'].str.rstrip('(').str.replace('&amp;', '&', regex=False)
        
        return items_df
    
    def desc_processing(self,  items_df: pd.DataFrame) -> pd.DataFrame:  
        items_df['description'] = items_df['description'].str[0]
        
        return items_df
    
    def convert_df_to_dict(self, items_df: pd.DataFrame) -> dict:  
        items_dict = items_df.set_index("asin").to_dict(orient="index")
        
        return items_dict
    
    def processing_flow(self, items_df: pd.DataFrame) -> dict:
        items_df_core = self.extract_7_core(items_df)
        items_df_clean = self.drop_duplicate(items_df_core)
        items_df_clean = self.price_processing(items_df_clean)
        items_df_clean = self.rank_processing(items_df_clean)
        items_df_clean = self.desc_processing(items_df_clean)
        items_df_clean = self.convert_df_to_dict(items_df_clean)
        
        return items_df_clean   

if __name__ == "__main__":
    # Ví dụ sử dụng
    pre = preProcessing_metadata()
    meta_data_dict = pre.processing_flow(meta_data_df)
    
    generator = ItemEmbeddingGenerator()
    generator.debug_prompt(meta_data_dict)
    